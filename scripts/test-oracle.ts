import { setupOracle } from './oracle-logic.js';

const main = async () => {
	console.log('🔮 Starting oracle...\n');
	const oracle = await setupOracle();

	const question = 'How much damage does a character take from a 30 foot fall?';

	console.log(`Question: ${question}\n`);

	const response = await oracle.invoke({ input: question });
	console.log(`Answer: ${response.answer}\n`);
};

main();
