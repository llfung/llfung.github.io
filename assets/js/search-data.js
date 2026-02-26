// get the ninja-keys element
const ninja = document.querySelector('ninja-keys');

// add the home and posts menu items
ninja.data = [{
    id: "nav-about",
    title: "about",
    section: "Navigation",
    handler: () => {
      window.location.href = "/";
    },
  },{id: "nav-projects",
          title: "projects",
          description: "All my research projects, past, present and future ideas.",
          section: "Navigation",
          handler: () => {
            window.location.href = "/projects/";
          },
        },{id: "nav-publications",
          title: "publications",
          description: "Publications are sorted in reversed chronological order.",
          section: "Navigation",
          handler: () => {
            window.location.href = "/publications/";
          },
        },{id: "nav-teaching",
          title: "teaching",
          description: "I can teach any 1st and 2nd year engineering courses, but my main teaching is in Fluids, Numerics and Machine Learning.",
          section: "Navigation",
          handler: () => {
            window.location.href = "/teaching/";
          },
        },{id: "nav-cv",
          title: "CV",
          description: "This is a brief CV following the standard set by jsonresume.org. For the complete version, follow the link to the PDF.",
          section: "Navigation",
          handler: () => {
            window.location.href = "/cv/";
          },
        },{id: "nav-blog",
          title: "blog",
          description: "",
          section: "Navigation",
          handler: () => {
            window.location.href = "/blog/";
          },
        },{id: "post-welcome-to-my-new-blog",
        
          title: "Welcome to my new blog!",
        
        description: "They say there is a time for everything - a time to be born and a time to die. I think my ideas should also have a place to live, or die.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/Welcome/";
          
        },
      },{id: "books-the-godfather",
          title: 'The Godfather',
          description: "",
          section: "Books",handler: () => {
              window.location.href = "/books/the_godfather/";
            },},{id: "news-odr-bindy-release",
          title: 'ODR-BINDy release!',
          description: "",
          section: "News",handler: () => {
              window.location.href = "/news/announcement_2/";
            },},{id: "news-for-the-coming-academic-year-i-am-going-to-organise-the-aerodynamics-and-control-seminar-series-at-the-department-of-aeronautics-of-imperial-college-london-do-get-in-touch-see-orcid-if-you-re-interested-in-coming-to-give-a-talk-on-relevant-subjects",
          title: 'For the coming academic year, I am going to organise the Aerodynamics and...',
          description: "",
          section: "News",},{id: "news-the-long-awaited-review-paper-on-continuum-modelling-of-active-suspensions-is-finally-published-in-the-themed-issue-biological-fluid-dynamics-emerging-directions-on-phil-tran-roc-soc-a-this-was-years-of-hard-work-summarising-over-decades-long-of-research-progress-and-experience-in-the-field-it-might-also-be-the-most-concise-derivation-of-almost-all-existing-continuum-models-for-active-suspensions-brag",
          title: 'The long-awaited review paper on continuum modelling of active suspensions is finally published...',
          description: "",
          section: "News",},{id: "news-the-workshop-on-symbolic-model-discovery-has-concluded-on-23-sept-2025-check-out-the-workshop-recording-the-tutorial-exercise-on-www-symbolicmodel-org",
          title: 'The workshop on symbolic model discovery has concluded on 23 Sept, 2025. Check...',
          description: "",
          section: "News",},{id: "news-i-ll-be-at-the-uk-ai-for-turbulence-workshop-at-the-british-library-london-on-26-27-january-2026-see-you-there",
          title: 'I’ll be at the UK AI for Turbulence Workshop at the British Library,...',
          description: "",
          section: "News",},{id: "news-i-ll-be-giving-a-talk-on-remember-to-forget-overcoming-the-ill-conditioned-inverse-problem-in-chaos-by-information-theory-at-the-3rd-ercoftac-ml4fluid-workshop-at-cwi-amsterdam-on-4-march-2026-i-hope-to-see-you-there",
          title: 'I’ll be giving a talk on “Remember to ‘forget’: overcoming the ill conditioned...',
          description: "",
          section: "News",},{id: "projects-adjoint-accelerated-programmable-inference-for-large-pdes",
          title: 'Adjoint-accelerated Programmable Inference for Large PDEs',
          description: "Enabling Adjoint in Probabilistic Programming Language Turing.jl",
          section: "Projects",handler: () => {
              window.location.href = "/projects/1_project/";
            },},{id: "projects-symbolic-model-discovery",
          title: 'Symbolic Model Discovery',
          description: "Discovering governing equation from temporal observation of the states",
          section: "Projects",handler: () => {
              window.location.href = "/projects/2_project/";
            },},{id: "projects-nn-vs-fem",
          title: 'NN vs FEM',
          description: "Which method is better at solving PDEs? To understand that, We need a deep dive into representation theory.",
          section: "Projects",handler: () => {
              window.location.href = "/projects/3_project/";
            },},{id: "projects-recovering-hidden-states-by-esn-bindy",
          title: 'Recovering Hidden States by ESN + BINDy',
          description: "Classical BINDy/SINDy require observation of all states. Can we improve upon that?",
          section: "Projects",handler: () => {
              window.location.href = "/projects/4_project/";
            },},{id: "projects-continuum-modelling-of-active-brownian-particles-abp",
          title: 'Continuum modelling of Active Brownian Particles (ABP)',
          description: "A better model than Pedley&#39;s 1992 classic and  Genearlized Taylor Dispersion, which does not generalise to arbitrary flow field.",
          section: "Projects",handler: () => {
              window.location.href = "/projects/5_project/";
            },},{id: "projects-gyrotactic-plume-formation-and-bioconvection",
          title: 'Gyrotactic plume formation and bioconvection',
          description: "A series of work on gyrotactic plume gives new interpretation on the classical pattern that is bioconvection.",
          section: "Projects",handler: () => {
              window.location.href = "/projects/6_project/";
            },},{id: "projects-colonial-choanoflagellates-and-choanocytes-in-sponges",
          title: 'Colonial Choanoflagellates and Choanocytes in sponges',
          description: "Does the newly found colonial Choanoflagellates shed light on the origin of multicellular life?",
          section: "Projects",handler: () => {
              window.location.href = "/projects/7_project/";
            },},{id: "projects-sedimentation-of-elongated-phytoplankton-in-the-ocean",
          title: 'Sedimentation of elongated phytoplankton in the ocean',
          description: "We found a singularity that focuses long, slender particle when they sink!",
          section: "Projects",handler: () => {
              window.location.href = "/projects/8_project/";
            },},{id: "projects-hydrodynamic-interactions-from-low-to-high-reynolds-number",
          title: 'Hydrodynamic interactions from low to high Reynolds number',
          description: "Suspension sedimentation, but not Stokesian at phenomenlogical scale.",
          section: "Projects",handler: () => {
              window.location.href = "/projects/9_project/";
            },},{
        id: 'social-github',
        title: 'GitHub',
        section: 'Socials',
        handler: () => {
          window.open("https://github.com/llfung", "_blank");
        },
      },{
        id: 'social-orcid',
        title: 'ORCID',
        section: 'Socials',
        handler: () => {
          window.open("https://orcid.org/0000-0002-1775-5093", "_blank");
        },
      },{
        id: 'social-scholar',
        title: 'Google Scholar',
        section: 'Socials',
        handler: () => {
          window.open("https://scholar.google.com/citations?user=JDoL5RMAAAAJ", "_blank");
        },
      },{
      id: 'light-theme',
      title: 'Change theme to light',
      description: 'Change the theme of the site to Light',
      section: 'Theme',
      handler: () => {
        setThemeSetting("light");
      },
    },
    {
      id: 'dark-theme',
      title: 'Change theme to dark',
      description: 'Change the theme of the site to Dark',
      section: 'Theme',
      handler: () => {
        setThemeSetting("dark");
      },
    },
    {
      id: 'system-theme',
      title: 'Use system default theme',
      description: 'Change the theme of the site to System Default',
      section: 'Theme',
      handler: () => {
        setThemeSetting("system");
      },
    },];
