import asyncio
from llama_index.core.workflow import (
    Context,
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step,
)
from rag_setup import query_engine, llm


class TranslatedEvent(Event):
    english_question: str
    language: str


class AnswerEvent(Event):
    english_question: str
    english_answer: str
    language: str


class DarwinWorkflow(Workflow):
    @step
    async def translate_question(self, ctx: Context, ev: StartEvent) -> TranslatedEvent:
        question = ev.question

        # read the memory saved by earlier questions
        history = await ctx.store.get("history", default=[])
        if history:
            history_text = "\n".join(
                "Q: " + h["question"] + "\nA: " + h["answer"] for h in history
            )
        else:
            history_text = "(none)"

        lang_reply = await llm.acomplete(
            "Which language is the following text written in? "
            "Reply with exactly one word: English, Dutch or Arabic.\n\n" + question
        )
        language = lang_reply.text.strip().rstrip(".")

        prompt = (
            "Conversation so far (in English):\n" + history_text + "\n\n"
            "New question: " + question + "\n\n"
            "Rewrite the new question as a standalone question in English. "
            "Translate it if needed, and replace references such as 'it', "
            "'that' or 'them' using the conversation so far. If it is already "
            "a standalone English question, repeat it unchanged. "
            "Reply with only the question."
        )
        eng_reply = await llm.acomplete(prompt)
        english = eng_reply.text.strip()
        print("Language:", language, "| Standalone English question:", english)
        return TranslatedEvent(english_question=english, language=language)

    @step
    async def search_book(self, ev: TranslatedEvent) -> AnswerEvent:
        response = await query_engine.aquery(ev.english_question)
        return AnswerEvent(
            english_question=ev.english_question,
            english_answer=str(response),
            language=ev.language,
        )

    @step
    async def translate_answer(self, ctx: Context, ev: AnswerEvent) -> StopEvent:
        # save this turn (in English) to the memory, keep the last 3 turns
        history = await ctx.store.get("history", default=[])
        history.append({"question": ev.english_question, "answer": ev.english_answer})
        await ctx.store.set("history", history[-3:])

        if ev.language.lower().startswith("english"):
            return StopEvent(result=ev.english_answer)
        reply = await llm.acomplete(
            "Translate the following text into " + ev.language + ". "
            "Reply with only the translation.\n\n" + ev.english_answer
        )
        return StopEvent(result=reply.text.strip())


async def main():
    w = DarwinWorkflow(timeout=120, verbose=False)
    ctx = Context(w)
    print("Ask about the book in English, Dutch or Arabic. Type exit to quit.")
    while True:
        question = input("\nYou: ").strip()
        if question.lower() in ("exit", "quit", ""):
            break
        result = await w.run(question=question, ctx=ctx)
        print("Answer:", result)


asyncio.run(main())
