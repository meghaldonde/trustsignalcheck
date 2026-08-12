(() => {
  // Data minimization: truncate at source to limit what leaves the browser
  // Matches backend limit (2000 chars) - no need to send extra data
  const MAX_TEXT_LENGTH = 2000;
  const text = document.body.innerText.trim().slice(0, MAX_TEXT_LENGTH);
  return {
    url: window.location.href,
    text: text,
  };
})();
