# POET Server Setup

In this guide, we will show you how to set up the POET ILP server for use locally or as a hosted service. We use this to host our POET server that powers requests from the [POET Demo Colab](https://colab.research.google.com/drive/1iup_edJd9zB1tfVBHXLmkWOT5yoSmXzz?usp=sharing) notebook. You do not need to set-up a POET-server to use POET, but feel free to use it to set-up your hosted service. 

POET's Integer Linear Program (ILP) formulation is compatible with a variety of solvers, including Gurobi and COIN-OR CBC. In this guide, we will demonstrate how to set it up using both of these solvers.

## Setting Up Gurobi (Optional)

The ILP solver defaults to using the COIN-OR CBC solver when Gurobi isn't available. However, since Gurobi is much faster, it is recommended to install it where possible.

### Acquiring a Free Academic Gurobi Web License

1. Create a free Gurobi account [here](https://pages.gurobi.com/registration). Make sure to specify the Academic user option.
2. Complete the rest of the Gurobi account creation process, which will include creating a password and verifying your email address.
3. Login to the Gurobi [Web License Manager](https://license.gurobi.com/) using your new account.
4. Create and download a new Web License file. This will be a `gurobi.lic` file that you will need in later steps, so keep note of where you save it.

## Option 1: Running the Server Locally

1. If using Gurobi, move the `gurobi.lic` file you downloaded in the previous step to your home directory (i.e. to `~/gurobi.lic`).
2. Clone this repository by running `git clone https://github.com/ShishirPatil/poet`.
3. Run `pip3 install -e .` in this repository's root directory to install the `poet-ai` package.
4. Run `cd poet-server` to navigate to the `poet-server` directory.
5. Run `pip3 install -r requirements.txt` to install the ILP server dependencies.
6. Finally, run `python3 server.py` to start the server.
    - You can optionally run `DEV=1 python3 server.py` to enable reload mode, which will automatically restart the server when you make changes to the code.
7. You can now make requests to the server at `http://localhost/solve`.

## Option 2: Running the Server Locally within a Docker Container

We include a Docker image that can be used to run the server. 

Prebuilt Docker images are available at `public.ecr.aws/i5z6k9k2/poet-server`

You can pull an image and start the server using:

```bash
docker pull public.ecr.aws/i5z6k9k2/poet-server:latest
docker run -p 80:80 -v ~/gurobi.lic:/opt/gurobi/gurobi.lic public.ecr.aws/i5z6k9k2/poet-server
```

Or, you can build the docker container yourself following the steps below. 


1. Ensure you have [Docker Compose](https://docs.docker.com/compose/install/) installed.
2. Clone this repository by running `git clone https://github.com/ShishirPatil/poet`.
3. If using Gurobi, move the `gurobi.lic` file you downloaded in the previous step to the `poet-server` directory of this repository (i.e. to `poet-server/gurobi.lic`).
4. Run `cd poet-server` to navigate to the `poet-server` directory.
5. Run `docker compose up --build` to build and start the Docker container.
6. You can now make GET requests to the server at `http://localhost/solve` as shown below.

## Option 3: Hosting POET server on an AWS EC2 Instance

Ensure that you have moved the `gurobi.lic` file (if you want to use the Gurobi optimizer) you downloaded earlier to the EC2 instance. Ensure that Port 80 is open for ingress traffic. 
 

## Making Requests 

To issue requests to the POET server, you can use the following Python code. Here, we use the demo POET-server hosted at IP `54.189.43.62`:

```python
import requests

response = requests.get("http://54.189.43.62/solve", {
    "model": "linear",
    "platform": "m0",
    "ram_budget": 90000000,
    "runtime_budget": 1.253,
    "solver": "gurobi",
})

print(response.json())
```



