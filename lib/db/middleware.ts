// Refreshes the Supabase session cookies on every request. Without
// this, expired access tokens would force the learner to sign in
// again mid-practice. Called from middleware.ts at the repo root.

import { NextResponse, type NextRequest } from "next/server";
import { createServerClient } from "@supabase/ssr";

import type { Database } from "./database.types";

export async function updateSession(request: NextRequest) {
  // Make the pathname available to server components via headers().
  // Used by SiteNav to decide whether to render (suppressed on the
  // landing page which carries its own nav). The header must be
  // forwarded through `NextResponse.next({ request: { headers } })`
  // for `headers()` in server components to see it — `request.headers.set`
  // alone does not propagate.
  const requestHeaders = new Headers(request.headers);
  requestHeaders.set("x-pathname", request.nextUrl.pathname);

  let response = NextResponse.next({ request: { headers: requestHeaders } });

  const supabase = createServerClient<Database>(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!,
    {
      cookies: {
        getAll() {
          return request.cookies.getAll();
        },
        setAll(cookiesToSet) {
          for (const { name, value } of cookiesToSet) {
            request.cookies.set(name, value);
          }
          response = NextResponse.next({ request: { headers: requestHeaders } });
          for (const { name, value, options } of cookiesToSet) {
            response.cookies.set(name, value, options);
          }
        },
      },
    },
  );

  // Calling getUser() refreshes the access token if it has expired,
  // which writes new cookies via the setAll callback above. The
  // returned value itself is not used here — downstream code reads
  // the user via lib/db/server.ts.
  await supabase.auth.getUser();

  return response;
}
