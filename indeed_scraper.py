import os
import smtplib
import spacy
import urllib3

from bs4 import BeautifulSoup
from email.message import EmailMessage
from sklearn.neighbors import NearestNeighbors
from dotenv import load_dotenv

# Load environment variables.
# Docker
# env_path = '/usr/src/.env'
# load_dotenv(dotenv_path=env_path, verbose=True)

# Local
load_dotenv()

PASSWORD = os.getenv('PASSWORD')
USER_NAME = os.getenv('USER_NAME')
print('usr ', USER_NAME)


class IndeedScraper(object):
    """Get Indeed.com job listings that match closest with your provided document.
    
    Use BeautifulSoup to scrape Indeed.com for job listings and descriptions.
    Use the Spacy NLP library to vectorize each listing and your provided document.
    Then, use KNN to find listings most relevant to your provided document.
    Get the results right in your email inbox!

    User supplies:
    - The number of indeed.com pages to search.
    - The number of search results to return.
    - The email address to send results to.
    - The file name of your provided document.
    - The city you want to work in.
    - The state that city is in.
    - A search term for the kind of job you're looking for (i.e. Data Scientist).

    """

    def __init__(self) -> None:
        self.pages = self.num_user_input('Enter number of pages to search:\n')  # Number of indeed pages to search.
        self.num_jobs = self.num_user_input('Enter max job listings to receive:\n') * 2  # Buffer for duplicates.
        self.email = self.user_input('Enter email:\n')
        self.city = self.user_input('Enter city:\n').strip().title()
        self.state = self.user_input('Enter state:\n').strip().upper()
        self.terms = self.user_input('Enter job title:\n').strip().lower()
        self.resume_path = self.user_input('Enter resume file name:\n')
        self.resume = self.load_resume(self.resume_path)
        self.url = self.build_url()
        self.http = urllib3.PoolManager()
        self.jobs = []
        self.base_email = 'pkutrich@gmail.com'
        self.vectors = None
        print('Loading NLP packages...')
        self.nlp = spacy.load("en_core_web_lg")
        self.nn = NearestNeighbors(n_neighbors=self.num_jobs,
                                   algorithm='ball_tree')
        self.descriptions = None
        self.main()
    
    def main(self) -> None:
        """Calls all methods needed to complete program."""
        self.descriptions = self.get_descriptions()
        self.vectors = self.get_description_vectors()
        self.get_best_jobs()
        self.email_jobs()
    
    def build_url(self) -> str:
        """Builds search url from user input."""
        url = f"http://www.indeed.com/jobs?q=" \
              f"{'%20'.join(self.terms.split())}&l={'%20'.join(self.city.split())},%20{self.state}"
        print('Search URL: ', url)
        return url

    @staticmethod
    def user_input(prompt: str) -> str:
        """Prompts user with <prompt> and returns string input."""
        return input(prompt)

    @staticmethod
    def num_user_input(prompt: str) -> int:
        """Prompts user with <prompt> and returns integer input."""
        while True:
            try:
                num = int(input(prompt))
                break
            except ValueError:
                print('Please enter a number.')
                continue

        return num

    @staticmethod
    def find_long_descriptions(soup) -> list:
        """Create list of urls for long form job descriptions."""
        urls = []
        for div in soup.find_all(name='div', 
                                 attrs={'class': 'row'}):
            for a in div.find_all(name='a', 
                                  attrs={'class': 'jobtitle turnstileLink'}):
                urls.append(a['href'])
        return urls

    def get_next_pages(self) -> list:
        """Create a list of top level pages to search."""
        return [self.url] + [self.url + f'&start={x}0' for x in range(1, self.pages)]

    def get_descriptions(self) -> list:
        """Create a list of text extracted from long form job descriptions."""
        print('Getting job descriptions...')
        descriptions = []
        # Get and parse each top level page.
        for base_url in self.get_next_pages():
            request = self.http.request('GET',
                                        base_url)
            base_soup = BeautifulSoup(request.data, "html.parser")
            # Follow links to each job description on the page.
            for url in self.find_long_descriptions(base_soup):
                the_url = "http://www.indeed.com/" + url
                req = self.http.request('GET', 
                                        the_url,
                                        headers={'User-Agent': 'opera'},
                                        retries=urllib3.Retry(connect=500, 
                                                              read=2,
                                                              redirect=50))
                # Parse out text from each description page and put it in the list.
                soup = BeautifulSoup(req.data, 'html.parser')
                description = soup.find(name='div', 
                                        attrs={'id': 'jobDescriptionText'})
                if description:
                    descriptions.append((the_url, description.text))
        print(f"Found {len(descriptions)} jobs.")
        return descriptions

    @staticmethod
    def load_resume(path) -> str:
        """Load resume text from disc."""
        print('\nLoading resume...')
        with open(path, 'r') as f:
            resume = f.read().strip('\n')
        return resume
    
    def get_description_vectors(self) -> list:
        """Get Spacy vectors for each long form job description."""
        print('Getting description vectors...')
        return [self.nlp(doc).vector for _, doc in self.descriptions]
        
    def get_best_jobs(self) -> None:
        """Vectorize resume and fit a nearest neighbors classifier to find desired number of jobs."""
        print(f'Finding best {self.num_jobs // 2} job matches...')
        self.nn.fit(self.vectors)
        neighbors = list(self.nn.kneighbors([self.nlp(self.resume).vector], self.num_jobs, return_distance=False)[0])
        for neighbor in neighbors:
            self.jobs.append(self.descriptions[neighbor])
        self.remove_duplicates()

    def remove_duplicates(self) -> None:
        """Use Spacy's similarity function to weed out duplicate job descriptions."""
        final_jobs = [self.jobs[0]]
        for job in self.jobs[1:]:
            doc1 = self.nlp(job[1])
            # Compare the similarity of <job> to each doc in <final_jobs>.
            # If <job> matches any of <final_jobs> reject <job>.
            if all([doc1.similarity(self.nlp(doc[1])) < .99 for doc in final_jobs]):
                final_jobs.append(job)
                # Don't include more jobs than were asked for.
                if len(final_jobs) == self.num_jobs // 2:
                    break
        self.jobs = final_jobs.copy()

    def email_jobs(self) -> None:
        """Send list of jobs to user."""
        print('Emailing jobs...')
        msg = self.build_message()
        server = self.initialize_server()
        self.send_and_deactivate(server, msg)

    def build_message(self):
        """Create EmailMessage instance."""
        msg = EmailMessage()
        msg['subject'] = "New jobs!!"
        msg['from'] = self.base_email
        msg['to'] = self.email
        div = "\n" + "*-" * 40 + "\n"
        msg.set_content(f"{div}".join([job[0].strip() + f'{div}' + job[1] + f'{div}' for job in self.jobs]))
        return msg

    @staticmethod
    def initialize_server():
        """Start a Gmail smtp server."""
        server = smtplib.SMTP('smtp.gmail.com', 587)
        # server.set_debuglevel(1)
        server.starttls()
        server.login(USER_NAME, PASSWORD)
        return server

    @staticmethod
    def send_and_deactivate(server, msg) -> None:
        """Send <msg> and deactivate <server>. Print confirmation message."""
        server.send_message(msg)
        server.quit()
        print("You've got mail!!")


if __name__ == "__main__":
    scraper = IndeedScraper()
