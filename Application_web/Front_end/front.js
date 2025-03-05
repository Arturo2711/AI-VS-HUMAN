document
  .getElementById("button_send_text")
  .addEventListener("click", request_model);

function updateBars(value1, value2) {
  const humanBar = document.getElementById("humanProbability");
  const machineBar = document.getElementById("machineProbability");
  if (humanBar) {
    humanBar.style.width = `${value1}%`;
    humanBar.innerText = `${value1.toFixed(2)}%`;
  }
  if (machineBar) {
    machineBar.style.width = `${value2}%`;
    machineBar.innerText = `${value2.toFixed(2)}%`;
  }
}

async function request_model() {
  const submitText = document.getElementById("submitText");
  const loadingSpinner = document.getElementById("loadingSpinner");

  try {
    submitText.innerText = "Loading...";
    loadingSpinner.classList.remove("d-none");

    const text_input = document.getElementById("inputText").value;

    const options = {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      mode: "cors",
      body: JSON.stringify({ text: text_input }),
    };

    const response = await fetch(
      "https://ai-vs-human-production.up.railway.app/request_model",
      options
    );

    if (!response.ok) {
      throw new Error(`Response status: ${response.status}`);
    }

    const response_json = await response.json();

    updateBars(100 * response_json.human, 100 * response_json.machine);
  } catch (error) {
    console.error("Error:", error.message);
  } finally {
    submitText.innerText = "Submit";
    loadingSpinner.classList.add("d-none");
  }
}
