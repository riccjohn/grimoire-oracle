import { Box, render, Text } from 'ink';
import TextInput from 'ink-text-input';
import { useEffect, useState } from 'react';
import { setupOracle } from './oracle-logic';
import { theme } from './theme';
import { AIMessage, HumanMessage } from '@langchain/core/messages';

const crystal_ball = '\udb82\udf2f';

type Message = { role: 'human' | 'ai'; content: string };

const App = () => {
  const [oracle, setOracle] = useState<Awaited<
    ReturnType<typeof setupOracle>
  > | null>(null);

  const [query, setQuery] = useState('');
  const [status, setStatus] = useState<'idle' | 'loading' | 'error'>('idle');
  const [messages, setMessages] = useState<Message[]>([]);

  useEffect(() => {
    setupOracle().then(setOracle);
  }, []);

  const handleSubmit = async () => {
    if (!oracle || !query.trim()) {
      setStatus('error');
      return;
    }

    // Add user message immediately (optimistic update)
    const userMessage: Message = { role: 'human', content: query };
    setMessages((prev) => [...prev, userMessage]);
    setQuery('');
    setStatus('loading');

    // Convert our messages to LangChain format for chat_history
    const chatHistory = messages.map((m) =>
      m.role === 'human'
        ? new HumanMessage(m.content)
        : new AIMessage(m.content),
    );
    const response = await oracle.invoke({
      input: query,
      chat_history: chatHistory,
    });

    // Add AI response
    const aiMessage: Message = { role: 'ai', content: response.answer };
    setMessages((prev) => [...prev, aiMessage]);
    setStatus('idle');
  };

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
              m.role === 'human' ? theme.userResponse : theme.oracleResponse
            }
          >
            {m.role === 'human' ? '❯ ' : '🧙 '}
            {m.content}
          </Text>
        ))}
      </Box>
      {status === 'loading' && (
        <Text italic color={theme.loading}>
          Consulting the grimoire...
        </Text>
      )}
      {!oracle && <Text color={theme.loading}>Loading oracle...</Text>}
      {status === 'idle' && oracle && (
        <>
          <TextInput
            value={query}
            onChange={setQuery}
            onSubmit={handleSubmit}
            placeholder="Ask me about OSE rules..."
          />
        </>
      )}
    </Box>
  );
};

render(<App />);
