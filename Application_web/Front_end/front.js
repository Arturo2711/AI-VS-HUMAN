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
  try {
    const text_input = document.getElementById("inputText").value;
    const options = {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      mode: "cors",
      body: JSON.stringify({ text: text_input }),
    };

    const response = await fetch("https://aihumantext.netlify.app", options);

    if (!response.ok) {
      throw new Error(`Response status: ${response.status}`);
    }
    const response_json = await response.json();
    /// Now, use probabilities to update probability bars
    console.log(`human ${response_json.human}`);
    console.log(`machine ${response_json.machine}`);
    updateBars(100 * response_json.human, 100 * response_json.machine);
  } catch (error) {
    console.error(error.message);
  }
}
