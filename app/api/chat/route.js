import { NextResponse } from "next/server";
import { Pinecone } from "@pinecone-database/pinecone";
import OpenAI from "openai";

const systemPrompt = `
You are a rate my professor agent to help students find classes, that takes in user questions and answers them.
For every user question, the top 3 professors that match the user question are returned.
Use them to answer the question if needed.
`;

export async function POST(req) {
    try {
        const data = await req.json();
        
        // Initialize Pinecone and OpenAI clients with environment variables
        const pc = new Pinecone({
            apiKey: process.env.PINECONE_API_KEY,
        });
        
        const index = pc.index('rag').namespace('ns1');
        const openai = new OpenAI({
            apiKey: process.env.OPENAI_API_KEY,
        });

        const text = data[data.length - 1].content;

        // Generate text embedding
        const embedding = await openai.embeddings.create({
            model: 'text-embedding-3-small',
            input: text,
        });

        // Query Pinecone index
        const results = await index.query({
            topK: 3,
            includeMetadata: true,
            vector: embedding.data[0].embedding,
        });

        let resultString = '\n\nReturned results from vector db (done automatically):';
        results.matches.forEach((match) => {
            resultString += `
            Returned Results:
            Professor: ${match.id}
            Review: ${match.metadata.review}
            Subject: ${match.metadata.subject}
            Stars: ${match.metadata.stars}\n\n
            `;
        });

        const lastMessage = data[data.length - 1];
        const lastMessageContent = lastMessage.content + resultString;
        const lastDataWithoutLastMessage = data.slice(0, data.length - 1);

        // Create a chat completion
        const completion = await openai.chat.completions.create({
            model: "gpt-4o-mini",
            messages: [
                { role: "system", content: systemPrompt },
                ...lastDataWithoutLastMessage,
                { role: 'user', content: lastMessageContent },
            ],
            stream: true,
        });

        // Stream the response
        const stream = new ReadableStream({
            async start(controller) {
                const encoder = new TextEncoder();
                try {
                    for await (const chunk of completion) {
                        const content = chunk.choices[0]?.delta?.content;
                        if (content) {
                            controller.enqueue(encoder.encode(content));
                        }
                    }
                } catch (err) {
                    console.error("Error while streaming response:", err); // Log streaming error
                    controller.error(err);
                } finally {
                    controller.close();
                }
            },
        });

        return new NextResponse(stream);
    } catch (error) {
        console.error("Error occurred during the request:", error); // Log the error details
        return new NextResponse(`Internal Server Error: ${error.message}`, { status: 500 });
    }
}
