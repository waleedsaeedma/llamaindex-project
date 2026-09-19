from llama_index.core.tools import QueryEngineTool
from rag_setup import query_engine

tool = QueryEngineTool.from_defaults(
    query_engine=query_engine,
    name="origin_of_species_search",
    description=(
        "Searches Charles Darwin's On the Origin of Species. "
        "Use it to answer any question about the book. "
        "The input must be a question written in English."
    ),
)

print("Tool name:", tool.metadata.name)
print("Description:", tool.metadata.description)

result = tool.call("What does Darwin say about the struggle for existence?")
print(result.content)
