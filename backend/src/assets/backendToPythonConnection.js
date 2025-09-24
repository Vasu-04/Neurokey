const express = require('express');
const { spawn } = require('child_process');

const router = express.Router();
const path = require('path');
const pythonScriptPath = path.join(__dirname, '../Python/keystrokeAnalyzer.py'); // Adjust the path as needed
class BackendToPythonConnection {
    async runPythonScript(input) {
        // console.log(input)
        return new Promise((resolve, reject) => {
            const pythonProcess = spawn('python', [pythonScriptPath]);
            let scriptOutput = '';

            pythonProcess.stdout.on('data', (data) => {
                console.log("PYTHON STDOUT:", data.toString());
                scriptOutput += data.toString();
            });
            pythonProcess.stderr.on('data', (data) => {
                console.error(`Error: ${data}`);
            });

            pythonProcess.on('close', (code) => {
                // console.log(`Python script finished with code ${code}`);
                if (code === 0){
                    try{
                        const result = JSON.parse(scriptOutput);
                        // console.log("Parsed script output:", result);
                        resolve(result);
                    }
                    catch (err){
                        reject(`Failed to parse script output: ${err}`);
                    }
                }
                else {
                    reject(`Python script exited with code ${code}`);
                }
            });
            // console.log("Input to Python Script: ", input);
            pythonProcess.stdin.write(JSON.stringify({ input }));
            pythonProcess.stdin.end();

        });
    }
}

module.exports = BackendToPythonConnection;