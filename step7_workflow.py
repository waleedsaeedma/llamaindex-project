import asyncio
from llama_index.core.workflow import Event, StartEvent, StopEvent, Workflow, step
from rag_setup import query_engine, llm


class TranslatedEvent(Event):
    english_question: str
    language: str


class AnswerEvent(Event):
    english_answer: str
    language: str


class DarwinWorkflow(Workflow):
    @step
    async def translate_question(self, ev: StartEvent) -> TranslatedEvent:
        question = ev.question
        lang_reply = await llm.acomplete(
            "Which language is the following text written in? "
            "Reply with exactly one word: English, Dutch or Arabic.\n\n" + question
        )
        language = lang_reply.text.strip().rstrip(".")
        eng_reply = await llm.acomplete(
            "Translate the following question into English. Reply with only the "
            "translation. If it is already in English, repeat it unchanged.\n\n"
            + question
        )
        english = eng_reply.text.strip()
        print("Language:", language, "| English question:", english)
        return TranslatedEvent(english_question=english, language=language)

    @step
    async def search_book(self, ev: TranslatedEvent) -> AnswerEvent:
        response = await query_engine.aquery(ev.english_question)
        return AnswerEvent(english_answer=str(response), language=ev.language)

    @step
    async def translate_answer(self, ev: AnswerEvent) -> StopEvent:
        if ev.language.lower().startswith("english"):
            return StopEvent(result=ev.english_answer)
        reply = await llm.acomplete(
            "Translate the following text into " + ev.language + ". "
            "Reply with only the translation.\n\n" + ev.english_answer
        )
        return StopEvent(result=reply.text.strip())


async def main():
    w = DarwinWorkflow(timeout=120, verbose=False)
    print("Ask about the book in English, Dutch or Arabic. Type exit to quit.")
    while True:
        question = input("\nYou: ").strip()
        if question.lower() in ("exit", "quit", ""):
            break
        result = await w.run(question=question)
        print("Answer:", result)


asyncio.run(main())
