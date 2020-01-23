# Job-finder
Docker container with CLI to scrape Indeed.com, vectorize resume and job descriptions using Spacy, and find jobs relevant to resume using KNN. Email results.

## Usage
- Save resume in `.txt` format.
- Remove any non-ascii characters and bullets from `<your resume>.txt` file.
- Consider removing your personal info from `<your resume>.txt` like name and location info for more generalizable results.
- Start Docker app in terminal.
    - `$ docker run -it pkutrich/job-finder`
- *Don't start entering info just yet!*
- In a separate terminal:
    - `$ cd` to directory containing your `<your resume>.txt`
        - `$ docker ps`
        - copy `NAMES` of the container running the `IMAGE` `pkutrich/job-finder`
        - `$ docker cp <your resume>.txt <name of running container>:/usr/src/<your resume>.txt`
- Return to first terminal and follow prompts.
- Program will print status updates.
- Check your email!
    