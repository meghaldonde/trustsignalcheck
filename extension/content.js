(() => {
  // Extract the article body, not the page.
  //
  // The previous version took document.body.innerText.slice(0, 2000). On a
  // typical article page the first 2,000 characters in DOM order are the nav,
  // the cookie banner, the newsletter promo and the share rail -- roughly 80%
  // page chrome. Scans came back with the model describing "navigation
  // scaffolding" because that is genuinely most of what it was given.
  //
  // Strategy, in order:
  //   1. <article>, then <main>, then common content containers
  //   2. otherwise, the block with the highest paragraph-text density
  //   3. otherwise, fall back to body text (previous behaviour)
  // Boilerplate elements are removed from a working clone first, so a match on
  // <main> still drops a <nav> nested inside it.

  const MAX_TEXT_LENGTH = 2000;   // matches the backend limit

  const STRIP = [
    "nav", "header", "footer", "aside", "form", "script", "style", "noscript",
    "svg", "button", "iframe", "figure > figcaption", "[role=navigation]",
    "[role=banner]", "[role=contentinfo]", "[role=complementary]",
    "[aria-hidden=true]", "[hidden]",
  ].join(",");

  const NOISE_HINT = /(^|[-_ ])(nav|menu|cookie|consent|banner|promo|newsletter|subscribe|share|social|sidebar|related|comment|advert|ad-|breadcrumb|skip)/i;

  const CANDIDATES = [
    "article", "main", "[role=main]", "[itemprop=articleBody]",
    ".post-body", ".article-body", ".entry-content", ".post-content",
    "#content", ".content",
  ];

  const clean = (node) => {
    const copy = node.cloneNode(true);
    copy.querySelectorAll(STRIP).forEach((el) => el.remove());
    // Drop containers whose class/id look like chrome (cookie bars, promos).
    copy.querySelectorAll("[class],[id]").forEach((el) => {
      if (NOISE_HINT.test(el.className || "") || NOISE_HINT.test(el.id || "")) {
        el.remove();
      }
    });
    return copy;
  };

  const textOf = (node) => (node.innerText || node.textContent || "")
    .replace(/[ \t ]+/g, " ")
    .replace(/\n{3,}/g, "\n\n")
    .trim();

  // Score a block by how much of it is prose. Paragraph text is weighted so a
  // list of nav links loses to a shorter run of real sentences.
  const score = (node) => {
    const all = textOf(node).length;
    if (all < 200) return 0;
    let para = 0;
    node.querySelectorAll("p, li, blockquote, h1, h2, h3").forEach((p) => {
      const t = textOf(p);
      if (t.length > 40) para += t.length;   // ignore link-sized fragments
    });
    return para * (para / Math.max(all, 1));  // density-weighted prose length
  };

  let best = null;
  let bestScore = 0;

  for (const sel of CANDIDATES) {
    for (const node of document.querySelectorAll(sel)) {
      const c = clean(node);
      const s = score(c);
      if (s > bestScore) { best = c; bestScore = s; }
    }
  }

  // No named container won -- scan generic blocks for the densest prose.
  if (bestScore === 0) {
    for (const node of document.querySelectorAll("body div, body section")) {
      if (node.querySelector("div, section")) continue;  // leaf-ish blocks only
      const c = clean(node);
      const s = score(c);
      if (s > bestScore) { best = c; bestScore = s; }
    }
  }

  let text = best ? textOf(best) : "";
  let source = best ? "article" : "body";

  // Last resort: cleaned body, then raw body. Pages that are genuinely lists
  // (search results, product grids) legitimately land here.
  if (text.length < 200) {
    text = textOf(clean(document.body));
    source = "body-cleaned";
  }
  if (text.length < 50) {
    text = textOf(document.body);
    source = "body-raw";
  }

  const payload = text.slice(0, MAX_TEXT_LENGTH);

  // Logged to the page console so extraction can be inspected on real sites.
  // This is how you capture the exact input the model receives -- open DevTools
  // on the page, click Scan, then copy the logged string. Useful while tuning
  // extraction; safe to delete once it stops earning its keep.
  console.log(`[SignalCheck] extraction=${source} chars=${payload.length}\n${payload}`);

  return {
    url: window.location.href,
    text: payload,
    // Reported so the admin dashboard can show when extraction fell back --
    // a spike in "body-*" means the strategy is failing on real pages.
    extraction: source,
  };
})();
