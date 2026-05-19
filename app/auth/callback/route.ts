// OAuth callback for Google and the magic-link confirmation hop.
// Supabase Auth sends the user here after they complete the external
// flow; we exchange the `code` for a session and forward them to
// the page they came from (or the home page).

import { NextResponse, type NextRequest } from "next/server";

import { createClient } from "@/lib/db/server";

export async function GET(request: NextRequest) {
  const url = new URL(request.url);
  const code = url.searchParams.get("code");
  const next = url.searchParams.get("next") ?? "/";

  if (code) {
    const supabase = await createClient();
    const { error } = await supabase.auth.exchangeCodeForSession(code);
    if (!error) {
      return NextResponse.redirect(new URL(next, request.url));
    }
  }

  // Code missing or exchange failed — bounce back to the sign-in
  // screen with a flag so the UI can surface the error to the user.
  return NextResponse.redirect(new URL("/sign-in?error=callback", request.url));
}
