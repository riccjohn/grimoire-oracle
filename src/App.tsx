import { AIMessage, HumanMessage } from "@langchain/core/messages";
import { Box, render, Text, useInput } from "ink";
import { useEffect, useRef, useState } from "react";
import { setupOracle } from "./oracle-logic";
import { theme } from "./theme";

const crystal_ball = "\udb82\udf2f";

type Message = { role: "human" | "ai"; content: string };

const App = () => {
	const [oracle, setOracle] = useState<Awaited<
		ReturnType<typeof setupOracle>
	> | null>(null);

	const [query, setQuery] = useState("");
	const queryRef = useRef("");
	const [status, setStatus] = useState<"idle" | "loading" | "error">("idle");
	const [messages, setMessages] = useState<Message[]>([]);

	useEffect(() => {
		setupOracle().then(setOracle);
	}, []);

	const handleSubmit = async (submittedQuery: string) => {
		if (!oracle || !submittedQuery.trim()) {
			setStatus("error");
			return;
		}

		// Add user message immediately (optimistic update)
		const userMessage: Message = { role: "human", content: submittedQuery };
		setMessages((prev) => [...prev, userMessage]);
		setStatus("loading");

		// Convert our messages to LangChain format for chat_history
		const chatHistory = messages.map((m) =>
			m.role === "human"
				? new HumanMessage(m.content)
				: new AIMessage(m.content),
		);
		const response = await oracle.invoke({
			input: submittedQuery,
			chat_history: chatHistory,
		});

		// Add AI response
		const aiMessage: Message = { role: "ai", content: response.answer };
		setMessages((prev) => [...prev, aiMessage]);
		setStatus("idle");
	};

	const isInputActive = status === "idle" && oracle !== null;

	useInput(
		(input, key) => {
			if (key.return) {
				const submittedQuery = queryRef.current;
				queryRef.current = "";
				setQuery("");
				handleSubmit(submittedQuery);
			} else if (key.backspace || key.delete) {
				queryRef.current = queryRef.current.slice(0, -1);
				setQuery(queryRef.current);
			} else if (!key.ctrl && !key.meta && input) {
				queryRef.current += input;
				setQuery(queryRef.current);
			}
		},
		{ isActive: isInputActive },
	);

	return (
		<Box
			flexDirection="column"
			padding={1}
			borderStyle="round"
			borderColor={theme.oracleTitle}
		>
			<Text bold color={theme.oracleTitle}>
				{crystal_ball} THE GRIMOIRE ORACLE {crystal_ball}
			</Text>
			<Box flexDirection="column" marginY={1}>
				{messages.slice(-6).map((m, i) => (
					<Text
						key={i}
						color={
							m.role === "human" ? theme.userResponse : theme.oracleResponse
						}
					>
						{m.role === "human" ? "❯ " : "🧙 "}
						{m.content}
					</Text>
				))}
			</Box>
			{status === "loading" && (
				<Text italic color={theme.loading}>
					Consulting the grimoire...
				</Text>
			)}
			{!oracle && <Text color={theme.loading}>Loading oracle...</Text>}
			{status === "idle" && oracle && (
				<Text>
					{query.length > 0 ? (
						query
					) : (
						<Text dimColor>Ask me about OSE rules...</Text>
					)}
					<Text backgroundColor="white"> </Text>
				</Text>
			)}
		</Box>
	);
};

render(<App />);
