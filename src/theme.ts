/**
 * Catppuccin Mocha palette for the Grimoire Oracle TUI.
 * https://catppuccin.com/palette
 *
 * Usage: <Text color={theme.oracle}>Oracle says...</Text>
 */
import { flavors } from '@catppuccin/palette';

const mocha = flavors.mocha.colors;

export const theme = {
  // Oracle (AI responses)
  oracleTitle: mocha.lavender.hex,
  oracleResponse: mocha.sapphire.hex,

  // User (input & messages)
  userTitle: mocha.green.hex,
  userResponse: mocha.teal.hex,

  // UI states
  dim: mocha.overlay0.hex,
  error: mocha.red.hex,
  success: mocha.green.hex,
  loading: mocha.yellow.hex,
  accent: mocha.peach.hex,

  // Direct palette access (if needed)
  palette: mocha,
};
