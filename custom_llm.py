from langchain.llms.base import LLM
from typing import List, Any, Dict, Tuple
from langchain.chains import ChatVectorDBChain


def _get_chat_history(chat_history: List[Tuple[str, str]]) -> str:
    buffer = ""
    for human_s, ai_s in chat_history:
        human = "Human: " + human_s
        ai = "Assistant: " + ai_s
        buffer += "\n" + "\n".join([human, ai])
    return buffer


class ChatVectorDBWithPineconeMetadataFilterChain(ChatVectorDBChain):
    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        print("inputs", inputs)
        filter = inputs["filter"]
        question = inputs["question"]
        get_chat_history = self.get_chat_history or _get_chat_history
        chat_history_str = get_chat_history(inputs["chat_history"])
        vectordbkwargs = inputs.get("vectordbkwargs", {})
        if chat_history_str:
            new_question = self.question_generator.run(
                question=question, chat_history=chat_history_str
            )
        else:
            new_question = question
        docs = self.vectorstore.similarity_search(
            new_question,
            k=self.top_k_docs_for_context,
            filter=filter,
            **vectordbkwargs,
        )
        # print("docs", docs)
        new_inputs = inputs.copy()
        new_inputs["question"] = new_question
        new_inputs["chat_history"] = chat_history_str
        answer, _ = self.combine_docs_chain.combine_docs(docs, **new_inputs)
        if self.return_source_documents:
            return {self.output_key: answer, "source_documents": docs}
        else:
            return {self.output_key: answer}
