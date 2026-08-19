# ACLED reference articles

The ACLED benchmark evaluates DataSTORM against 20 published ACLED analyses.
Each article is the human-written reference for one topic: the system is asked a
neutral research question, and its report is scored against the corresponding
article.

**The article texts are not redistributed here.** They are ACLED's copyrighted
publications. This directory lists the exact source for each one so you can
retrieve them yourself under ACLED's terms.

## Sources

| # | Article | Published |
|---|---|---|
| 1 | [Peace talks in Nariño may be a litmus test for Petro’s bid to end Colombia’s conflict](https://acleddata.com/report/peace-talks-narino-may-be-litmus-test-petros-bid-end-colombias-conflict) | 12 December 2024 |
| 2 | [“Total Peace” paradox in Colombia: Petro’s policy reduced violence, but armed groups grew stronger](https://acleddata.com/report/total-peace-paradox-colombia-petros-policy-reduced-violence-armed-groups-grew-stronger) | 28 November 2024 |
| 3 | [Conflict intensifies and instability spreads beyond Burkina Faso, Mali, and Niger](https://acleddata.com/report/conflict-intensifies-and-instability-spreads-beyond-burkina-faso-mali-and-niger) | 12 December 2024 |
| 4 | [Russia’s protracted war on Ukraine may be reaching a turning point](https://acleddata.com/report/russias-protracted-war-ukraine-may-be-reaching-turning-point) | 12 December 2024 |
| 5 | [Al-Shabaab targets civilians in Somalia in retaliation for installing CCTV cameras - November](https://acleddata.com/report/al-shabaab-targets-civilians-somalia-retaliation-installing-cctv-cameras-november-2024) | 29 November 2024 |
| 6 | [Two years after the Pretoria agreement, unrest still looms in Tigray - October](https://acleddata.com/update/two-years-after-pretoria-agreement-unrest-still-looms-tigray-october-2024) | 8 November 2024 |
| 7 | [Georgia: An “existential” election](https://acleddata.com/report/georgia-existential-election) | 21 October 2024 |
| 8 | [Artillery shelling and airstrikes surge in Sudan - September](https://acleddata.com/report/artillery-shelling-and-airstrikes-surge-sudan-september-2024) | 16 September 2024 |
| 9 | [In Amhara, over 7 million people are exposed to political violence - August](https://acleddata.com/update/amhara-over-7-million-people-are-exposed-political-violence-august-2024) | 13 September 2024 |
| 10 | [Militants thrive amid political instability in Pakistan](https://acleddata.com/report/militants-thrive-amid-political-instability-pakistan) | 12 December 2024 |
| 11 | [Between cooperation and competition: The struggle of resistance groups in Myanmar](https://acleddata.com/report/between-cooperation-and-competition-struggle-resistance-groups-myanmar) | 26 November 2024 |
| 12 | [Kenya battles threats from communal militias and al-Shabaab - November](https://acleddata.com/report/kenya-battles-threats-communal-militias-and-al-shabaab-november-2024) | 25 November 2024 |
| 13 | [Cabo Ligado Update: 11 - 24 November](https://acleddata.com/update/cabo-ligado-update-11-24-november-2024) | 28 November 2024 |
| 14 | [A year after SNNPR’s dissolution, violence returns to historically troubled areas - September 2024](https://acleddata.com/update/year-after-snnprs-dissolution-violence-returns-historically-troubled-areas-september-2024) | 17 October 2024 |
| 15 | [Viv Ansanm: Living together, fighting united — the alliance reshaping Haiti’s gangland](https://acleddata.com/report/viv-ansanm-living-together-fighting-united-alliance-reshaping-haitis-gangland) | 16 October 2024 |
| 16 | [Foreign meddling and fragmentation fuel the war in Sudan](https://acleddata.com/report/foreign-meddling-and-fragmentation-fuel-war-sudan) | 12 December 2024 |
| 17 | [Mexico’s new administration braces for shifting battle lines in the country’s gang wars](https://acleddata.com/report/mexicos-new-administration-braces-shifting-battle-lines-countrys-gang-wars) | 12 December 2024 |
| 18 | [The Rwanda Defence Force (RDF) operations abroad signal a shift in Rwanda’s regional standing](https://acleddata.com/report/rwanda-defence-force-rdf-operations-abroad-signal-shift-rwandas-regional-standing) | 27 September 2024 |
| 19 | [Ethiopia Weekly Update (3 December 2024)](https://acleddata.com/update/ethiopia-weekly-update-3-december-2024) | 5 December 2024 |
| 20 | [Defection and violence against civilians in Sudan’s al-Jazirah state - November](https://acleddata.com/report/defection-and-violence-against-civilians-sudans-al-jazirah-state-november-2024) | 18 November 2024 |

Article IDs match the reference file names the evaluators expect
(`references/<id>.md`) and the topic numbering used throughout `results/`. The
same list, with each article's type and the neutral research prompt derived from
it, is Appendix E of [the paper](https://arxiv.org/abs/2604.06474).

## Obtaining the texts

To reproduce the ACLED evaluation, save each article as `references/<id>.md`
with the article title on the first line and the body below it — the format the
evaluators expect. `references/*.md` is gitignored, so retrieved texts cannot be
committed back by accident.

## License

These articles are published by the Armed Conflict Location & Event Data Project
(ACLED) and are subject to ACLED's terms of use and attribution requirements,
not to this repository's Apache-2.0 license. Review
<https://acleddata.com/terms-of-use/> before downloading, storing or
redistributing them, and cite ACLED as the source in any derived work. The same
applies to the underlying ACLED event data, which we also do not redistribute.
