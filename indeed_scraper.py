import numpy as np
import smtplib
import spacy
import urllib3

from bs4 import BeautifulSoup
from email.message import EmailMessage
from sklearn.neighbors import NearestNeighbors


class IndeedScraper(object):
    """Get Indeed.com job listings that match closest with your resume.
    
    Use BeautifulSoup to scrape Indeed.com for job listings and 
    descriptions.
    Use the Spacy NLP library to vectorize each listing and your resume,
    then use KNN to find listings most relevant to your resume. Get the 
    results right in your email inbox!

    You supply:
    - The email address to send results to.
    - The file name of your resume (in the same directory as this file).
    - The city you want to work in.
    - The state that city is in.
    - A search term for the kind of job you're looking for (i.e. Apprentice Plumber).

    :param pages: int number of indeed.com pages to search.
    :param num_jobs: int number of search results to return.

    """

    def __init__(self, 
                 pages: int = 10, 
                 num_jobs: int = 10,
                 email: str = None,
                 resume_path: str = None,
                 city: str = None,
                 state: str = None,
                 terms: str = None) -> None:
        self.email = self.user_input('Enter email:\n')
        self.city = self.user_input('Enter city:\n').strip().title()
        self.state = self.user_input('Enter state:\n').strip().upper()
        self.terms = self.user_input('Enter job title:\n').strip().lower()
        self.resume_path = self.user_input('Enter resume file name:\n')
        self.resume = self.load_resume(self.resume_path)
        self.url = self.build_url()
        self.http = urllib3.PoolManager()
        self.pages = pages
        self.num_jobs = num_jobs
        self.jobs = set()
        self.base_email = 'pkutrich@gmail.com'
        self.vectors = None
        print('Loading NLP packages...')
        self.nlp = spacy.load("en_core_web_lg")
        self.nn = NearestNeighbors(n_neighbors=self.num_jobs,
                                   algorithm='ball_tree')
        self.descriptions = None
        self.main()
    
    def main(self) -> None:
        self.descriptions = self.get_descriptions()
        self.vectors = self.get_description_vectors()
        self.get_best_jobs()
        self.email_jobs()
    
    def build_url(self) -> None:
        url = f"http://www.indeed.com/jobs?q={'%20'.join(self.terms.split())}&l={'%20'.join(self.city.split())},%20{self.state}"
        print('Search URL: ', url)
        return url
    
    def user_input(self, prompt: str) -> str:
        return input(prompt)

    def find_long_urls(self, soup: str) -> list:
        urls = []
        for div in soup.find_all(name='div', 
                                 attrs={'class': 'row'}):
            for a in div.find_all(name='a', 
                                  attrs={'class': 'jobtitle turnstileLink'}):
                urls.append(a['href'])
        return urls

    def get_next_pages(self) -> None:
        return [self.url] + [self.url + f'&start={x}0' for x in range(1, self.pages)]

    def get_descriptions(self) -> list:
        print('Getting job descriptions...')
        descriptions = []
        for base_url in self.get_next_pages():
            request = self.http.request('GET',
                                        base_url)
            base_soup = BeautifulSoup(request.data, 
                                      features="html.parser")

            for url in self.find_long_urls(base_soup):
                the_url = "http://www.indeed.com/" + url

                req = self.http.request('GET', 
                                        the_url,
                                        headers={'User-Agent': 'opera'},
                                        retries=urllib3.Retry(connect=500, 
                                                              read=2,
                                                              redirect=50))

                soup = BeautifulSoup(req.data, 'html.parser')
                description = soup.find(name='div', 
                                        attrs={'id': 'jobDescriptionText'})
                if description:
                    descriptions.append((the_url, description.text))
        print(f"Found {len(descriptions)} jobs.")
        return descriptions
    
    def load_resume(self, path) -> str:
        print('Loading resume...')
        with open(path, 'r') as f:
            resume = f.read().strip('\n')
        return resume
    
    def get_description_vectors(self):
        print('Getting description vectors...')
        return np.array([self.nlp(doc).vector for _, doc in self.descriptions])
        
    def get_best_jobs(self) -> None:
        print(f'Finding best {self.num_jobs} job matches...')
        self.nn.fit(self.vectors)
        potential_neighbors = self.nn.kneighbors(np.array([self.nlp(self.resume).vector]))
        neighbors = [y for x, y in zip(potential_neighbors[0][0], potential_neighbors[1][0]) if 0.05 < x]
        for neighbor in neighbors:
            self.jobs.add(self.descriptions[neighbor][1])
    
    def email_jobs(self) -> None:
        print('Emailing jobs...')
        msg = EmailMessage()
        msg['subject'] = "New jobs!!"
        msg['from'] = self.base_email
        msg['to'] = self.email
        div = "\n" + "-" * 79 + "\n"
        msg.set_content(f"{div}".join([job.strip() + '\n' for job in self.jobs]))
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login('pkutrich', 'ozgndzvnrgihyawj')
#         server.set_debuglevel(1)
        server.send_message(msg)
        server.quit()
        print("You've got mail!!")


if __name__ == "__main__":
    scraper = IndeedScraper(10, 10)
    
