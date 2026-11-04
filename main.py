import logging
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from deepagents.backends import StateBackend
from langchain_classic.chains.llm_summarization_checker.base import PROMPTS_DIR
from langchain_core.messages import HumanMessage

from generation.agents.agent import create_agent
from generation.agents.validation_agent import create_company_claim_validator
from generation.tools.search import search_content
from src.indexing import *

def main():
    load_dotenv()

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = Path("logs") / run_timestamp
    run_dir.mkdir(parents=True, exist_ok=False)

    run_logger = logging.getLogger(f"rag_run.{run_timestamp}")
    run_logger.setLevel(logging.INFO)
    run_logger.propagate = False
    file_handler = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    run_logger.addHandler(file_handler)
    run_logger.info("Run started")


    k = int(os.getenv('RETRIEVAL_K'))
    if k is None:
        k = 5

    config = EmbeddingConfig(
        'openai',
        openai_api_key= os.getenv('OPENAI_API_KEY'),
        openai_base_url= os.getenv('OPENAI_BASE_URL'),
        openai_model= os.getenv('OPENAI_MODEL'),

        bedrock_region_name= os.getenv('BEDROCK_REGION'),
        bedrock_profile_name= os.getenv('BEDROCK_PROFILE'),
        bedrock_model_id=os.getenv('BEDROCK_MODEL_ID')
    )
    embedding = get_embeddings(config)

    docs = load_context_docs("data/")
    docs = split_docs(docs)
    print(docs)
    run_logger.info("Loaded and split %d candidate document chunks", len(docs))

    retriever = create_retriever(docs, embedding, k=k)

    company_name = "Nestle"
    role_title = "Graduate 2027 - Digital Development"
    job_description = r"""
Shape the future through scientific breakthroughs and innovation.

As a Digital Food Safety Early Warning Associate Specialist at Nestlé Research, your mission is to enable the proactive detection of potential food safety issues by strengthening the evaluation of external scientific signals and the evolution of digital early warning systems.. You’ll make a difference by collaborating closely with food safety, quality, research, technology, and external data partners. This is your opportunity to turn ideas into sustainable and innovative solutions for people, pets, and the planet.

How you’ll make an impact as a Digital Food Safety Early Warning Associate Specialist

Strengthening proactive food safety early warning by scouting and evaluating emerging issues across priority contaminant areas.
Identifying and assessing internal and external data sources that improve the detection, prioritization, and scientific evaluation of early signals.
Contribute to the evaluation and deployment of digital technologies that enhance data-driven detection of emerging food safety topics.
Testing prototypes that help transform early warning processes into scalable digital solutions.
Participate in scientific collaborations that connect research, technology, and business needs to protect consumers.
Delivering project activities with focus, rigor, and follow-through to support impactful food safety decisions.
What will make you successful

Bringing basic chemistry knowledge and a good understanding of chemical contaminants, including how they can emerge, evolve, and impact food safety.
Applying knowledge of food safety and quality processes in the context of supply chains, incident occurrence, issue evolution, and potential consumer impact.
Using data and digital technologies in a food safety context, ideally to support data-driven risk assessment, root-cause analysis, and early warning decision-making.
Collaborating effectively with diverse internal and external partners, while translating research into practical business impact.
Communicating fluently in English; working proficiency in French is a plus.
Bringing collaboration, fresh thinking, bold questions, respect, and a positive, wellbeing-oriented mindset.
    """,

    retrieved_context = retriever.invoke(
        f"{job_description}\n\nFind the most relevant candidate experiences based on this job description, "
        "skills, projects, and achievements"
    )
    run_logger.info("Retrieved %d candidate context chunks", len(retrieved_context))

    candidate_evidence = "\n".join([
        f"- {doc.page_content}..."
        for doc in (retrieved_context if isinstance(retrieved_context, list) else [retrieved_context])
    ])

    (run_dir / "RAG_context.txt").write_text(candidate_evidence, encoding="utf-8")
    run_logger.info("Saved retrieved context")

    # The below tools cannot be created because the current architecture loads context immediately into context window, instead of allowing the agent to delegate and control chunking retrieval. This is a limitation of the current implementation and will be addressed in future iterations.
    # retriever_tool = retrieve_content
    # bound_search = search_content.bind(retriever=retriever)
    PROMPTS: dict[str, str] = {}

    for path in Path('src/generation/prompts/').iterdir():
        if not path.is_dir():
            var_name = path.name
            content = path.read_text(encoding="utf-8")
            PROMPTS[var_name] = content

    SYSTEM_PROMPT = (
        # + "\n\n"
        # + "=" * 80
        # + "\n\n" +
        PROMPTS["sys_cover_letter.txt"].format()
    )

    internet_search = {"type": "web_search"}
    agent = create_agent(SYSTEM_PROMPT, [internet_search])


    HUMAN_QUERY = PROMPTS["human_prompt.txt"].format(
        job_description=job_description,
        company_name=company_name,
        role_title=role_title,
        retrieved_context=candidate_evidence,
    )

    result = agent.invoke(
        {"messages": [HumanMessage(content=HUMAN_QUERY)]}
    )

    cover_letter = next(
        (msg.text for msg in reversed(result.get("messages", [])) if msg.text),
        None,
    )
    if cover_letter is None:
        raise RuntimeError("The cover-letter agent returned no text.")

    print(cover_letter)
    (run_dir / f"{company_name.replace(' ', '_')}.txt").write_text(
        cover_letter, encoding="utf-8"
    )
    run_logger.info("Saved generated cover letter")

    validator = create_company_claim_validator()
    validation_prompt = PROMPTS["company_claim_validation.txt"].format(
        company_name=company_name,
        role_title=role_title,
        cover_letter=cover_letter,
    )
    validation_result = validator.invoke(
        {"messages": [HumanMessage(content=validation_prompt)]}
    )
    validation_report = next(
        (
            msg.text
            for msg in reversed(validation_result.get("messages", []))
            if msg.text
        ),
        None,
    )
    if validation_report is None:
        raise RuntimeError("The company-claim validator returned no report.")

    (run_dir / "company_claim_validation.txt").write_text(
        validation_report, encoding="utf-8"
    )
    run_logger.info("Saved company claim validation report")
    run_logger.info("Run completed successfully")



if __name__ == "__main__":
    main()