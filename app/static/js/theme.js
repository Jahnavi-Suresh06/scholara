// Complete theme system
(function () {
    "use strict";
    const KEY = "theme";
    const media = window.matchMedia("(prefers-color-scheme: dark)");

    function preference() {
        return localStorage.getItem(KEY) || "system";
    }

    function actualTheme(value) {
        if (value === "system") {
            return media.matches ? "dark" : "light";
        }
        return value === "dark" ? "dark" : "light";
    }

    function apply(value) {
        const actual = actualTheme(value);
        document.documentElement.setAttribute("data-theme", actual);
        document.documentElement.style.colorScheme = actual;
    }

    function syncRadios() {
        const saved = preference();
        document.querySelectorAll('input[name="theme"]').forEach(radio => {
            radio.checked = radio.value === saved;
        });
    }

    apply(preference());

    document.addEventListener("DOMContentLoaded", () => {
        syncRadios();
        document.querySelectorAll('input[name="theme"]').forEach(radio => {
            radio.addEventListener("change", function () {
                if (!this.checked) return;
                localStorage.setItem(KEY, this.value);
                apply(this.value);
                syncRadios();
            });
        });
    });

    media.addEventListener("change", () => {
        if (preference() === "system") apply("system");
    });
})();
