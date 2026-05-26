// Server-side Supabase client. Carries the user's session via cookies so
// RLS policies fire against the authenticated user. Use this in server
// components, server actions, and route handlers.

import { cookies } from "next/headers";
import { createServerClient } from "@supabase/ssr";

import type { Database } from "./database.types";

export async function createClient() {
  const cookieStore = await cookies();

  return createServerClient<Database>(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!,
    {
      cookies: {
        getAll() {
          return cookieStore.getAll();
        },
        setAll(cookiesToSet) {
          try {
            for (const { name, value, options } of cookiesToSet) {
              cookieStore.set(name, value, options);
            }
          } catch {
            // Server Components cannot set cookies; ignore. The auth
            // session refresh happens on the next mutation path.
          }
        },
      },
    },
  );
}

type ServerClient = Awaited<ReturnType<typeof createClient>>;

// Resilient wrapper around supabase.auth.getUser(). The call hits the Supabase
// auth endpoint over the network and can intermittently fail with a transient
// "fetch failed"; an uncaught throw during a server render shows the global
// error boundary (the intermittent crash, digest 3063673210). This retries the
// transient failure briefly and returns null rather than throwing, so callers
// degrade to the unauthenticated path (redirect to sign-in) instead of crashing.
export async function getUser(supabase: ServerClient, tries = 3) {
  for (let attempt = 0; attempt < tries; attempt++) {
    try {
      const {
        data: { user },
      } = await supabase.auth.getUser();
      return user;
    } catch (err) {
      if (attempt === tries - 1) {
        console.error("supabase.auth.getUser failed after retries", err);
        return null;
      }
      await new Promise((resolve) => setTimeout(resolve, 150 * (attempt + 1)));
    }
  }
  return null;
}
