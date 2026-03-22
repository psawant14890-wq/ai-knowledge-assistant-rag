from openai import OpenAI


class KnowledgeAssistant:
    def __init__(self, api_key, ingestor):
        self.client = OpenAI(api_key=api_key)
        self.ingestor = ingestor

    def ask(self, question, chat_history, answer_style="Balanced"):
        retrieved_docs = self.ingestor.retrieve(question, limit=5)
        history_text = self._history_to_text(chat_history)
        context = self._format_retrieved_docs(retrieved_docs)
        citations = self._extract_citations(retrieved_docs)

        prompt = f"""You are an expert knowledge assistant.
Answer the user's question using the retrieved context whenever possible.
If the context does not contain the answer, say so clearly and then provide the most helpful next step.
Never fabricate facts, citations, or source names.
Use the requested answer style: {answer_style}.

Conversation history:
{history_text or "No previous conversation."}

Retrieved context:
{context or "No relevant context was retrieved."}

Question:
{question}
"""
        response = self._generate_text(prompt)
        return {"response": response, "citations": citations}

    def summarize_knowledge_base(self):
        summary_chunks = self.ingestor.get_summary_documents(limit=12)
        if not summary_chunks:
            return "No knowledge sources are loaded yet."

        context = "\n\n".join(
            f"Source: {chunk['metadata']['source_name']}\n"
            f"Location: {chunk['metadata']['location']}\n"
            f"Content: {chunk['content']}"
            for chunk in summary_chunks
        )
        prompt = f"""You are summarizing a knowledge base built from PDFs and web pages.
Write a practical summary with:
1. The main topics covered.
2. The most important facts or ideas.
3. Any notable gaps or limitations in the sources.
4. Two suggested follow-up questions a user should ask next.

Knowledge base content:
{context}
"""
        return self._generate_text(prompt)

    def summarize_chat(self, chat_history):
        if not chat_history or len(chat_history) <= 1:
            return "No conversation history is available to summarize yet."

        prompt = f"""Summarize this conversation between a user and an AI assistant.
Include:
1. The user's main goals and questions.
2. The assistant's key answers.
3. Any unresolved items.
4. The most useful next actions.

Conversation:
{self._history_to_text(chat_history, limit=20)}
"""
        return self._generate_text(prompt)

    def suggest_questions(self):
        suggestion_chunks = self.ingestor.get_summary_documents(limit=8)
        if not suggestion_chunks:
            return []

        context = "\n\n".join(
            f"Source: {chunk['metadata']['source_name']}\nContent: {chunk['content'][:500]}"
            for chunk in suggestion_chunks
        )
        prompt = f"""You are helping a user explore a knowledge base.
Based on the sources below, suggest exactly 4 short, useful questions a user could ask next.
Return one question per line without numbering.

Sources:
{context}
"""
        text = self._generate_text(prompt)
        suggestions = [
            line.strip("- ").strip()
            for line in text.splitlines()
            if line.strip()
        ]
        return suggestions[:4]

    def _generate_text(self, prompt):
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,
            messages=[
                {
                    "role": "system",
                    "content": "You produce grounded, concise, helpful answers.",
                },
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content.strip()

    @staticmethod
    def _history_to_text(chat_history, limit=12):
        if not chat_history:
            return ""

        return "\n".join(
            f"{message['role'].title()}: {message['content']}"
            for message in chat_history[-limit:]
        )

    @staticmethod
    def _format_retrieved_docs(documents, max_chars=8000):
        chunks = []
        total_chars = 0
        for document in documents:
            metadata = document["metadata"]
            chunk = (
                f"Source: {metadata.get('source_name', 'Unknown source')}\n"
                f"Location: {metadata.get('location', 'Unknown location')}\n"
                f"Chunk: {metadata.get('chunk_index', 'N/A')}\n"
                f"Content: {document['content']}"
            )
            if total_chars + len(chunk) > max_chars:
                break
            chunks.append(chunk)
            total_chars += len(chunk)
        return "\n\n".join(chunks)

    @staticmethod
    def _extract_citations(documents):
        citations = []
        seen = set()
        for document in documents:
            metadata = document["metadata"]
            citation_key = (
                metadata.get("source_name", "Unknown source"),
                metadata.get("location", "Unknown location"),
                metadata.get("chunk_index", "N/A"),
            )
            if citation_key in seen:
                continue
            seen.add(citation_key)
            citations.append(
                {
                    "source": citation_key[0],
                    "location": citation_key[1],
                    "chunk_index": citation_key[2],
                    "snippet": document["content"][:220].strip(),
                }
            )
        return citations
