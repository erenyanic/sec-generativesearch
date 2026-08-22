"use client";

// Application shell: top navigation bar shared by every authenticated page.
//
// `WelcomeGate` renders the shell only once the operator is authenticated,
// so this component is never visible to anonymous callers. Sign-out lives
// here (in the nav bar) rather than on `WelcomeGate` so unauthenticated
// pages do not carry an unnecessary button.

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useCallback, useState, type JSX, type ReactNode } from "react";

import { signOutUser } from "@/lib/user-vault";

interface NavItem {
  href: string;
  label: string;
}

const NAV_ITEMS: NavItem[] = [
  { href: "/dashboard", label: "Dashboard" },
  { href: "/filings", label: "Filings" },
  { href: "/ingest", label: "Ingest" },
  { href: "/rag", label: "Ask" },
  { href: "/chat", label: "Chat" },
  { href: "/providers", label: "Providers" },
];

interface AppShellProps {
  children: ReactNode;
}

export function AppShell({ children }: AppShellProps): JSX.Element {
  const pathname = usePathname();
  const [signingOut, setSigningOut] = useState(false);

  const handleSignOut = useCallback(async () => {
    if (signingOut) {
      return;
    }
    setSigningOut(true);
    // Drop the in-memory user vault FIRST (wipes the KEK + cleartext map
    // before any network hop) and revoke the backend user session via
    // DELETE /api/auth/session. `signOutUser` wipes local state before
    // it awaits the network, so a slow / failed call cannot leave the
    // cleartext keys reachable from a stale render.
    try {
      await signOutUser();
    } catch {
      // Best-effort — the in-memory state is already wiped.
    }
    // Tear down the operator (admin) session in lockstep. Best-effort:
    // the HttpOnly cookies expire on their own if this fails.
    try {
      await fetch("/api/admin/session", {
        method: "DELETE",
        credentials: "same-origin",
        cache: "no-store",
      });
    } catch {
      // Best-effort.
    }
    // Reload to re-trigger the WelcomeGate + LoginGate probes; the
    // server-side store entries are already revoked.
    //
    // The HARD navigation is load-bearing and must NOT become
    // `router.push()`. `user-vault.ts` holds the KEK and the decrypted
    // provider-key map in MODULE MEMORY; only a full document teardown
    // discards them. A soft client-side navigation keeps the module graph
    // alive, so the KEK would survive "sign out" in the tab's JS heap.
    // eslint-disable-next-line @next/next/no-location-assign-relative-destination -- see above: full teardown of in-memory vault state
    window.location.assign("/");
  }, [signingOut]);

  return (
    <div className="min-h-screen">
      <header className="border-b border-slate-200 bg-white">
        <nav
          aria-label="Primary"
          className="mx-auto flex max-w-6xl items-center justify-between gap-6 px-6 py-3"
        >
          <div className="flex items-center gap-6">
            <Link
              href="/dashboard"
              className="text-sm font-semibold tracking-tight text-slate-900"
            >
              SEC-GenerativeSearch
            </Link>
            <ul className="flex items-center gap-1">
              {NAV_ITEMS.map((item) => {
                const active =
                  pathname === item.href ||
                  pathname?.startsWith(`${item.href}/`);
                return (
                  <li key={item.href}>
                    <Link
                      href={item.href}
                      aria-current={active ? "page" : undefined}
                      className={
                        "rounded px-3 py-1 text-sm " +
                        (active === true
                          ? "bg-slate-900 text-white"
                          : "text-slate-700 hover:bg-slate-100")
                      }
                    >
                      {item.label}
                    </Link>
                  </li>
                );
              })}
            </ul>
          </div>
          <button
            type="button"
            onClick={() => {
              void handleSignOut();
            }}
            disabled={signingOut}
            className="rounded border border-slate-300 px-3 py-1 text-sm text-slate-700 hover:bg-slate-100 disabled:opacity-50"
          >
            {signingOut ? "Signing out…" : "Sign out"}
          </button>
        </nav>
      </header>
      <main className="mx-auto max-w-6xl px-6 py-8">{children}</main>
    </div>
  );
}
