# Transcribe a sensitive interview

In this exercise we use OpenAI Whisper to run a local speech-to-text tool to convert an audio interview into text. 

## Task
*Handle a sensitive interview so that it can be uploaded into a trusted research environment, and then in the environment secure virtual desktop, run whisper to transcribe speech into audio. Use the local word editor and sound player to verify that the transcription was successful. Pseudonymise / paraphrase / annotate the interview as you would do normally.*

## Notes

Make sure you have the tools installed (see for example the conda environment in this repository). 

First run

`whisper JohnChowning041306_part1.ogg --model_dir 


The original interview is released uder CC-SA and is available at https://upload.wikimedia.org/wikipedia/commons/f/f7/JohnChowning041306_part1.ogg 
