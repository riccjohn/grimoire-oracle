import { AIMessage, HumanMessage } from 'langchain';
import { setupOracle } from '../src/oracle-logic.js';

const main = async () => {
	console.log('🔮 Starting oracle...\n');
	const oracle = await setupOracle();

	const firstQuestion = 'Tell me about the Light Spell';
	console.log(`You: ${firstQuestion}`);

	const firstResponse = await oracle.invoke({
		input: firstQuestion,
		chat_history: [],
	});
	console.log(`Oracle: ${firstResponse.answer}\n\n`);

	const secondQuestion = 'How long does it last?';
	console.log(`You: ${secondQuestion}`);

	const secondResponse = await oracle.invoke({
		input: secondQuestion,
		chatHistory: [
			new HumanMessage(firstQuestion),
			new AIMessage(firstResponse.answer),
		],
	});

	console.log(`Oracle: ${secondResponse.answer}\n\n`);
};

main();
