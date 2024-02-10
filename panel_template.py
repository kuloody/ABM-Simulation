from mesa.visualization.ModularVisualization import VisualizationElement


class RightPanelElement(VisualizationElement):
    local_includes = ["RightPanelModule"]

    def __init__(self):
        super().__init__()
        self.js_code = self._generate_js_code()

    def render(self, model):
        panel_html_top = self._generate_model_panel_html(100, model)
        panel_html_bottom = self._generate_button_panel_html(275)

        return f"""
            {self._generate_script()}
            {self._generate_style()}
            {panel_html_top}
            {panel_html_bottom}
        """

    def _generate_js_code(self):
        return "elements.push(new RightPanelModule());"

    def _generate_model_panel_html(self, top, model):
        return f"""
        <div class='well' style='margin-top:{top}px; margin-left:5px; position: fixed; top: {top}px; right: 50px;'>
            <h4 class="label-54">Model Variables</h4>
            <table style="font-size: smaller;">
                <tr class="label-54">
                    <td style="padding: 3px; font-size: smaller;" class="variable">Learning Students :</td>
                    <td style="padding: 3px; font-size: smaller; font-weight:bold;" class="variable">{model.learning}</td>
                </tr>
                <tr style="height: 10px;"></tr>
                <tr class="label-54">
                    <td style="padding: 3px; font-size: smaller;" class="variable">Disruptive Students :</td>
                    <td style="padding: 3px; font-size: smaller; font-weight:bold;" class="variable">{model.distruptive}</td>
                </tr>
                <tr style="height: 10px;"></tr>
                <tr class="label-54">
                    <td style="padding: 3px; font-size: smaller;" class="variable">Current school day :</td>
                    <td style="padding: 3px; font-size: smaller; font-weight:bold;" class="variable">{model.schoolDay}</td>
                </tr>
            </table>
        </div>
        """

    def _generate_button_panel_html(self, top):
        button_html = self._generate_button_html()
        return f"""
            <div id="button-panel" class='well' style='margin-top:{top}px; margin-left:10px; position: fixed; top: {top}px; left: 150px; width:250px; height:190px; visibility: hidden;'>
                {button_html}
            </div>
        """

    def _generate_button_html(self):
        return """
            <div id="container">
                <label for="fileInput" id="fileInputLabel" class="button-71" role="button">Upload Dataset</label>
                <input type="file" id="fileInput" accept=".csv" style="display: none;">
                <button class="button-71" id="generateDatasetBtn" role="button">Generate Dataset</button>
                <button id="show-results-btn" class="button-71" role="button">Show Results</button>
            </div>
        """

    def _generate_script(self):
        return """
        <script src="https://cdnjs.cloudflare.com/ajax/libs/PapaParse/5.3.0/papaparse.min.js"></script>
            <script>
                const fileInput = document.getElementById('fileInput');
                fileInput.addEventListener('change', handleFileInputChange);
                function handleFileInputChange(event) {
                    const fileInput = event.target;
                    const chosenFile = fileInput.files[0];
                    const reader = new FileReader();
                    // Update label with selected file name
                    const fileInputLabel = document.getElementById('fileInputLabel');
                    fileInputLabel.textContent = chosenFile ? chosenFile.name : 'Upload Dataset';
                    reader.onload = function(event) {
                        const fileContent = event.target.result;
                        // Parse CSV to JSON
                        Papa.parse(fileContent, {
                            header: true,
                            complete: function(results) {
                                const jsonData = results.data;
                                // Pass jsonData to a function that sends it to the backend
                                sendDataToBackend(jsonData);
                            }
                        });
                    };
                    reader.readAsText(chosenFile);
                }

                const flaskHostname = '127.0.0.1';
                const flaskPort = '5000';
                
                function sendDataToBackend(jsonData) {
                    const uploadURL = `http://${flaskHostname}:${flaskPort}/upload`; // Construct the complete URL
                    fetch(uploadURL, {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json' // Set content type to JSON
                        },
                        body: JSON.stringify(jsonData) // Convert jsonData to JSON format
                    })
                    .then(response => {
                        if (response.ok) {
                            // Data sent successfully
                            console.log('Data sent successfully');
                        } else {
                            // Error handling
                            console.error('Error sending data');
                        }
                    })
                    .catch(error => {
                        console.error('Error:', error);
                    });
                }

                
                // Define the processDataset function
                function processDataset() {
                    // Send an HTTP GET request to the Flask server to generate the dataset
                    fetch(`http://${flaskHostname}:${flaskPort}/generate_dataset`)
                        .then(response => {
                            if (!response.ok) {
                                // If the response status is not OK (200-299), throw an error
                                throw new Error(`HTTP error! Status: ${response.status}`);
                            }
                            // No need to parse JSON since the server does not return any data
                            // Display alert here
                            alert('Dataset generated successfully!');
                        })
                        .catch(error => {
                            // If an error occurs during the fetch request, show an error message
                            console.error('Error generating dataset:', error);
                            alert('Error generating dataset');
                        });
                }
                
                // Add event listener to the "Generate Dataset" button
                const generateDatasetBtn = document.getElementById('generateDatasetBtn');
                generateDatasetBtn.addEventListener('click', processDataset);

                document.getElementById("show-results-btn").addEventListener("click", function() {
                    window.open("http://127.0.0.1:5000/show_results", "_blank");
                });
            </script>
            <script>
            window.addEventListener('scroll', function() {
                var container = document.getElementById('container');
                var buttonPanel = document.getElementById('button-panel');
                if (window.scrollY > 140) {  // Adjust 140 to the desired scroll position
                    container.style.visibility = 'visible';
                   
                } else {
                    container.style.visibility = 'hidden';
              
                }
            });
        </script>
        """

    def _generate_style(self):
        return """
            <style>
                /* CSS */
                .file-chosen-label {
                    margin-top: 10px; /* Adjust as needed */
                    font-size: 14px;
                    color: #333; /* Text color */
                }
                h4 {
                    font-family: 'Merriweather', serif;
                    color: #2B3036;
                }
                .spacer {
                    margin-right: 20px; /* Adjust the spacing as needed */
                }
                #container {
                    display: flex;
                    flex-direction: column;
                    align-items: center; /* Align items vertically */
                    justify-content: center; /* Align items horizontally */
                    height: 100%; /* Set height to fill the container */
                }

                .label-54 {
                    font-family: "Open Sans", sans-serif; /* Set the font family */
                    font-size: 16px; /* Set the font size */
                    letter-spacing: 1px; /* Add some letter spacing */
                    text-decoration: none; /* Remove any text decoration */
                    text-transform: uppercase; /* Transform text to uppercase */
                    color: #000; /* Set the text color */
                    cursor: pointer; /* Set the cursor to a pointer, indicating interactivity */
                    border: 3px solid; /* Add a solid border */
                    padding: 0.25em 0.5em; /* Add padding to the label */
                    box-shadow: 1px 1px 0px 0px, 2px 2px 0px 0px, 3px 3px 0px 0px, 4px 4px 0px 0px, 5px 5px 0px 0px; /* Add a box shadow */
                    position: relative; /* Set the position to relative */
                    user-select: none; /* Disable text selection */
                    -webkit-user-select: none; /* Disable text selection for WebKit browsers */
                    touch-action: manipulation; /* Enable touch manipulation */
                    pointer-events: none; /* Disable pointer events */
                    margin-bottom: 10px; /* Adjust the value as needed for the desired spacing */
                }

                .label-54:active {
                    box-shadow: 0px 0px 0px 0px; /* Remove box shadow on click */
                    top: 5px; /* Move the label down slightly on click */
                    left: 5px; /* Move the label right slightly on click */
                }

                @media (min-width: 768px) {
                    .label-54 {
                        padding: 0.25em 0.75em; /* Adjust padding for larger screens */
                    }
                }

                .button-71 {
                    padding: 6px 12px; /* Adjust padding to reduce space */
                    width: auto;
                    background-color: #0078d0;
                    border: 0;
                    border-radius: 56px;
                    color: #fff;
                    cursor: pointer;
                    display: inline-block;
                    font-family: system-ui,-apple-system,system-ui,"Segoe UI",Roboto,Ubuntu,"Helvetica Neue",sans-serif;
                    font-size: 14px; /* Reduced font size */
                    font-weight: 600;
                    outline: 0;
                    position: relative;
                    text-align: center;
                    text-decoration: none;
                    transition: all .3s;
                    user-select: none;
                    -webkit-user-select: none;
                    touch-action: manipulation;
                    width: 220px; 
                    margin-bottom: 10px; /* Decreased margin-bottom */
                    height: 55px; /* Decreased button height */
                    padding: 4px 20px; /* Adjusted padding */
                }

                .button-71:before {
                    background-color: initial;
                    background-image: linear-gradient(#fff 0, rgba(255, 255, 255, 0) 100%);
                    border-radius: 125px;
                    content: "";
                    height: 50%;
                    left: 4%;
                    opacity: .5;
                    position: absolute;
                    top: 0;
                    transition: all .3s;
                    width: 92%;
                }

                .button-71:hover {
                    box-shadow: rgba(255, 255, 255, .2) 0 3px 15px inset, rgba(0, 0, 0, .1) 0 3px 5px, rgba(0, 0, 0, .1) 0 10px 13px;
                    transform: scale(1.05);
                }

                @media (min-width: 768px) {
                    .button-71 {
                        padding: 8px 48px; /* Adjusted padding */
                    }
                }
            </style>
        """
