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
        "Always translate the user's question into English before calling the "
        "origin_of_species_search tool. Answer only from what the tool returns. "
        "Write your final answer in the same language as the user's question."
    ),
)


async def ask(question, ctx):
    print("\nQuestion:", question)
    handler = agent.run(question, ctx=ctx)
    async for ev in handler.stream_events():
        if isinstance(ev, ToolCallResult):
            print("Tool call:", ev.tool_name, ev.tool_kwargs)
    response = await handler
    print("Answer:", response)


async def main():
    ctx = Context(agent)
    await ask("Wat is natuurlijke selectie volgens Darwin?", ctx)
    await ask("Can you give an example of it from the book?", ctx)


asyncio.run(main())
