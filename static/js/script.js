document.addEventListener("DOMContentLoaded", function () {
    const form = document.getElementById("predictionForm");
    const predictButton = document.getElementById("predictButton");
    const resetButton = document.getElementById("resetButton");

    if (form && predictButton) {
        form.addEventListener("submit", function () {
            predictButton.disabled = true;
            predictButton.textContent = "Sedang memproses prediksi...";
        });
    }

    if (form && resetButton) {
        resetButton.addEventListener("click", function (event) {
            const confirmed = window.confirm(
                "Apakah kamu yakin ingin mereset formulir?"
            );

            if (!confirmed) {
                event.preventDefault();
                return;
            }

            // Mengembalikan halaman ke kondisi awal setelah reset.
            window.setTimeout(function () {
                window.location.href = "/";
            }, 0);
        });
    }
});