(() => {
  const text = document.body.innerText.trim().slice(0, 5000);
  return {
    url: window.location.href,
    text: text,
  };
})();
