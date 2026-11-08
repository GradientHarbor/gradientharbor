// gradientharbor.com — hackathon landing page.
// The old BI landing page lives under /bi and is served by the separate
// "gradientharbor" Worker (GradientHarbor/website repo), routed on gradientharbor.com/bi*.
// This Worker serves everything else and redirects the old BI paths so existing links keep working.

const LEGACY_PREFIXES = ['/blog', '/docs'];

export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    // www → apex
    if (url.hostname === 'www.gradientharbor.com') {
      url.hostname = 'gradientharbor.com';
      return Response.redirect(url.toString(), 301);
    }

    // Old BI blog and docs URLs → /bi/...
    if (LEGACY_PREFIXES.some((p) => url.pathname === p || url.pathname.startsWith(p + '/'))) {
      url.pathname = '/bi' + url.pathname;
      return Response.redirect(url.toString(), 301);
    }

    return env.ASSETS.fetch(request);
  },
};
