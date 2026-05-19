import { defineConfig, globalIgnores } from "eslint/config";
import nextVitals from "eslint-config-next/core-web-vitals";
import nextTs from "eslint-config-next/typescript";

const eslintConfig = defineConfig([
  ...nextVitals,
  ...nextTs,
  // Override default ignores of eslint-config-next.
  globalIgnores([
    // Default ignores of eslint-config-next:
    ".next/**",
    "out/**",
    "build/**",
    "next-env.d.ts",
    // Python training subtree — not TypeScript, plus its venv ships
    // third-party JS (matplotlib) that we should not lint.
    "training/**",
    // Dataset + run outputs.
    "dataset/**",
    "runs/**",
    "artifacts/**",
    // Generated; format-check still applies via prettier.
    "lib/db/database.types.ts",
  ]),
]);

export default eslintConfig;
