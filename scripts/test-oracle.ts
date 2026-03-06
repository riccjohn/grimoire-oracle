import { AIMessage, HumanMessage } from "langchain";
import { setupOracle } from "../src/oracle-logic.js";

const main = async () => {
	console.log("🔮 Starting oracle...\n");
	const oracle = await setupOracle();

	const firstQuestion = "Tell me about the Thief class";
	console.log(`You: ${firstQuestion}`);

	const firstResponse = await oracle.invoke({
		input: firstQuestion,
		chat_history: [],
	});
	console.log(`Oracle: ${firstResponse.answer}\n\n`);

	const secondQuestion = "What skills do they have?";
	console.log(`You: ${secondQuestion}`);

	const secondResponse = await oracle.invoke({
		input: secondQuestion,
		chat_history: [
			new HumanMessage(firstQuestion),
			new AIMessage(firstResponse.answer),
		],
	});

	console.log(`Oracle: ${secondResponse.answer}\n\n`);
};

main();
