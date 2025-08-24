// vitest.workspace.ts
import { defineWorkspace } from 'vitest/config';

export default defineWorkspace([
  {
    name: 'client',
    extends: './vite.config.ts',
    test: {
      environment: 'browser',
      browser: {
        enabled: true,
        provider: 'playwright',
        instances: [{ browser: 'chromium' }]
      },
      include: ['src/**/*.{test,spec}.{js,ts}'],
      exclude: ['src/lib/server/**'],
      setupFiles: ['./vitest-setup-client.ts']
    }
  },
  {
    name: 'server',
    extends: './vite.config.ts',
    test: {
      environment: 'node',
      include: ['src/**/*.{test,spec}.{js,ts}'],
      exclude: ['src/**/*.svelte.{test,spec}.{js,ts}']
    }
  }
]);
