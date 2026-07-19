function copyBibTeX() {
  const bibtexElement = document.getElementById("bibtex-code");
  const button = document.querySelector(".copy-bibtex-btn");
  const copyText = button?.querySelector(".copy-text");

  if (!bibtexElement || !button || !copyText) {
    return;
  }

  const markCopied = () => {
    button.classList.add("copied");
    copyText.textContent = "Cop";
    window.setTimeout(() => {
      button.classList.remove("copied");
      copyText.textContent = "Copy";
    }, 2000);
  };

  navigator.clipboard.writeText(bibtexElement.textContent)
    .then(markCopied)
    .catch(() => {
      const textArea = document.createElement("textarea");
      textArea.value = bibtexElement.textContent;
      document.body.appendChild(textArea);
      textArea.select();
      document.execCommand("copy");
      textArea.remove();
      markCopied();
    });
}

function scrollToTop() {
  window.scrollTo({ top: 0, behavior: "smooth" });
}

window.addEventListener("scroll", () => {
  const scrollButton = document.querySelector(".scroll-to-top");
  if (scrollButton) {
    scrollButton.classList.toggle("visible", window.scrollY > 300);
  }
});

document.addEventListener("DOMContentLoaded", () => {
  if (window.bulmaCarousel) {
    window.bulmaCarousel.attach(".carousel", {
      slidesToScroll: 1,
      slidesToShow: 1,
      loop: true,
      infinite: true,
      autoplay: true,
      autoplaySpeed: 5000,
    });
  }
});
