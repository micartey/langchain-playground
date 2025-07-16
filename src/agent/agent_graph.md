---
config:
  flowchart:
    curve: linear
---
graph TD;
	__start__([<p>__start__</p>]):::first
	retrive_dfki_projects(retrive_dfki_projects)
	generate(generate)
	__end__([<p>__end__</p>]):::last
	__start__ -. &nbsp;default&nbsp; .-> generate;
	__start__ -. &nbsp;retrive&nbsp; .-> retrive_dfki_projects;
	retrive_dfki_projects --> generate;
	generate --> __end__;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc
