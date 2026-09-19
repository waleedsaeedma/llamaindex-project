import asyncio
from llama_index.core.agent.workflow import AgentWorkflow, ToolCallResult
from llama_index.core.tools import QueryEngineTool
from llama_index.core.workflow import Context
from rag_setup import query_engine, llm

query_engine_tool = QueryEngineTool.from_defaults(
    query_engine=query_engine,
    name="origin_of_species_search",
    description=(
        "Searches Charles Darwin's On the Origin of Species. "
        "Use it to answer any question about the book. "
        "The input must be a question written in English."
    ),
)

agent = AgentWorkflow.from_tools_or_functions(
    [query_engine_tool],
    llm=llm,
    system_prompt=(
        "You are an assistant that answers questions about Charles Darwin's "
        "On the Origin of Species. Users may write in English, Dutch or Arabic. "
        "Follow these steps for EVERY user message: "
        "1) Identify the language of the user's latest message only. Ignore the "
        "language of earlier messages and earlier answers. "
        "2) Translate the question into English and call the "
        "origin_of_species_search tool. Answer only from what the tool returns. "
        "3) Write the final answer in the language identified in step 1. "
        "If the latest message is in English, the answer must be in English, "
        "even if the previous answers were in Dutch or Arabic."
    ),
)


async def ask(question, ctx):
    handler = agent.run(question, ctx=ctx)
    async for ev in handler.stream_events():
        if isinstance(ev, ToolCallResult):
            print("Tool call:", ev.tool_name, ev.tool_kwargs)
    response = await handler
    print("Answer:", response)


async def main():
    ctx = Context(agent)
    print("Ask about the book in English, Dutch or Arabic. Type exit to quit.")
    while True:
        question = input("\nYou: ").strip()
        if question.lower() in ("exit", "quit", ""):
            break
        await ask(question, ctx)


asyncio.run(main())
