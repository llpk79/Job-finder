# Job-finder
Write a job description for your ideal job, (or use your resume) and use the power of Natural Language Processing (NLP) to search job sites for your ideal position so you can spend your time preparing for interviews instead of searching through listings!

Best results are expected by providing a job description writen in the style of a job description. The NLP algorithm is looking at document similarity as a whole, not simply searching for keywords or phrases.

Docker container runs on any machine. ((What is Docker?)[https://docs.docker.com/engine/docker-overview/])

After starting the program the command line prompts you to enter how many pages to scrape, how many jobs to return, where to search, what search phrase to use, and the name of your text file.

Currently searches only Indeed.com.

Python program uses:
- BeautifulSoup4 to scrape Indeed.com for job descriptions
- The Spacy NLP library to compare your ideal job description (or resume) to job descriptions found on Indeed, and find and remove duplicate listings
- A Scikit-Learn unsupervised machine learning technique call Nearest Neighbors to find the jobs best matching the document you provide
## Usage
- Save ideal job description (or resume) in `.txt` format
- If using resume, remove any non-ascii characters and bullets
- If using resume, consider removing your personal info, like name and location, for more generalizable results
- Start Job-finder in terminal
    - `$ docker run -it --rm pkutrich/job-finder`
- **Don't start entering info just yet!** Job-finder needs a copy of your document
- In a separate terminal
    - `$ cd` to directory containing your `<your_file>.txt`
        - `$ docker ps`
        - copy `NAMES` of the container running the `IMAGE` `pkutrich/job-finder`
        - `$ docker cp <your_file>.txt <name of running container>:<your_file>.txt`
- Return to first terminal and follow prompts
- Program will print status updates
- Check your email!
    