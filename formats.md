# GOV.UK Search API — Content Formats (HMRC)

**Source:** `https://www.gov.uk/api/search.json?count=1000&filter_organisations=hm-revenue-customs`
Retrieved 23 Sep 2026. The response contains `total: 100,164` results; the API returned the first 1,000, which contain 29 distinct `format` values.

Each result carries: `_id`, `title`, `description`, `link`, `format`, `document_type`, `es_score`, `organisations`, `public_timestamp`, and a few other metadata fields.

Counts below are from the 1,000-result sample, so they reflect what ranks early for this default query, not the full 100k corpus.

## Guidance content

### `guide` (184)
Mainstream user-facing guidance pages, typically task-oriented and often with child pages ("parts"). Examples: *HMRC online services: sign in or set up an account*, *Tax-Free Childcare*. Links live at top-level paths like `/log-in-register-hmrc-online-services`.

### `detailed_guide` (369)
Deeper technical/business guidance — the largest group in the sample. Examples: *Download the HMRC app*, *Check when you can expect a reply from HMRC*. The most common format for HMRC's substantial guidance collections (e.g. `/guidance/...` pages on Making Tax Digital, VAT, payroll).

### `guidance` (38)
Generic guidance publications that don't fit the guide/detailed_guide models — usually `/government/publications/...` documents such as lists and reference material. Examples: *List of approved professional organisations and learned societies (List 3)*, recognised Corporation Tax software suppliers.

## Manuals (the most interesting for this project)

### `hmrc_manual` (10)
The top-level HMRC internal manuals themselves — the cover page of each manual. Examples: *Capital Gains Manual*, *Pensions Tax Manual*, *Anti-money laundering guidance for supervised businesses*. Links follow `/hmrc-internal-manuals/<manual-slug>`.

### `hmrc_manual_section` (47)
Individual sections/pages **within** an HMRC internal manual, referenced by their manual page code in the title (e.g. `IEIM902330 - Tax Identification Number (TIN)`, `RFIG20500 - Statutory Residence Test...: Contents`). Links are `/hmrc-internal-manuals/<manual-slug>/<section-id>`. Note that "Contents" pages appear as sections too — these are the navigable index pages of a manual.

### `manual` (4) and `manual_section` (6)
A separate, older manual schema used for non-"internal-manual" manuals published as guidance. Examples: *The Tax Agent's Handbook* (`manual`) with sections like *Online services for agents* (`manual_section`), *Use Making Tax Digital for Income Tax*. These live under `/guidance/<slug>/<section>` rather than `/hmrc-internal-manuals/`.

**Implication:** to scrape full HMRC manuals, target `hmrc_manual` entries for the list of manuals and `hmrc_manual_section` (or the manual's own contents pages) for the body content — paging through the search API with `filter_format` would be needed, since this sample only surfaces a slice of the ~tens of thousands of manual sections.

## Interactive tools / transactions

### `transaction` (29)
Online services that require the user to sign in and do something — start a task, submit something. Examples: *Estimate your Income Tax*, *Check your Council Tax band*. Often front doors to HMRC digital services.

### `local_transaction` (2)
Like `transaction`, but the service is actually delivered by a local authority, with GOV.UK as an entry point. Examples: *Apply for Council Tax Reduction*, *Contact your council about business rates bill*.

### `smart_answer` (14)
Multi-question interactive tools that branch on user answers to produce a tailored outcome. Examples: *Calculate your statutory redundancy pay*, *Child Benefit tax calculator*.

### `simple_smart_answer` (8)
A single-question or simplified variant of a smart answer — usually "check if/what you need to do" flows. Examples: *Check if you need to send a Self Assessment tax return*, *Check how to register for Self Assessment*.

### `answer` (74)
Short, single-purpose pages answering one specific question — typically sign-in pages or quick factual answers. Examples: *Sign in to your childcare account*, *Personal tax account: sign in or set up*.

### `step_by_step_nav` (10)
Step-by-step navigable guides (the numbered "task list" UI pattern). Examples: *Set up a limited company: step by step*, *What to do when someone dies: step by step*.

### `form` (70)
Downloadable or fillable forms. Examples: *Inheritance Tax account (IHT400)*, CH459 Child Benefit form. Links point to `/government/publications/...` where the actual form (usually a PDF) is attached.

## Structured collections and finders

### `document_collection` (58)
A curated grouping of related documents under one landing page at `/government/collections/...`. Examples: *Paying HMRC: detailed information*, MTD for Income Tax step-by-step collections.

### `finder` (2)
Search/filter interfaces over a set of content. Examples: *Contact HM Revenue & Customs*, **Find HMRC manuals** (`/find-hmrc-manuals`) — this finder is itself the natural complement to the search API for enumerating manuals.

### `hmrc_contact` (46)
Individual contact pages for specific HMRC enquiry routes — phone numbers, addresses, opening hours per topic. Examples: *Income Tax: enquiries*, *Self Assessment: general enquiries*. One per entry in the contact finder.

## Corporate / news / policy

### `press_release` (13)
News announcements. Examples: State Pension awareness, Vaping Products Duty launch reminders. Path: `/government/news/...`.

### `news_story` (1)
Similar to a press release but a general news article (e.g. GOV.UK One Login announcement). Same `/government/news/...` path.

### `policy_paper` (3)
Formal policy documents — technical notes, Revenue & Customs Briefs. Examples: *Inheritance Tax on Pensions: Technical Note 2*, *Revenue and Customs Brief 10 (2026)*.

### `corporate_report` (2)
Transparency/official reports, e.g. *Named tax avoidance schemes*, *Details of deliberate tax defaulters*.

### `national_statistics` (1)
Official statistics releases (e.g. income distribution percentile tables under `/government/statistics/...`).

### `correspondence` (1)
Letters/newsletters published for the record, e.g. the MTD software developer newsletter.

### `international_treaty` (4)
Tax treaties and related documents between the UK and other states. Examples: *USA: tax treaties*, *Spain: tax treaties*.

## Organisation pages

### `organisation` (1)
The HMRC organisation homepage itself (`/government/organisations/hm-revenue-customs`).

### `about` (1)
The "About us" corporate page for the organisation.

### `recruitment` (1)
The organisation's recruitment page.

## Miscellaneous

### `promotional` (1)
Marketing-style content encouraging use of a service (e.g. *Use your HMRC business tax account*).

---

## Summary table

| Format | Count | What it is |
|---|---|---|
| detailed_guide | 369 | Deep technical/business guidance |
| guide | 184 | Mainstream task guidance |
| answer | 74 | Single-question answer pages |
| form | 70 | Downloadable forms |
| document_collection | 58 | Grouped document landing pages |
| hmrc_manual_section | 47 | Pages within an HMRC internal manual |
| hmrc_contact | 46 | Contact details per enquiry route |
| guidance | 38 | Other guidance publications |
| transaction | 29 | Sign-in online services |
| smart_answer | 14 | Multi-question interactive tools |
| press_release | 13 | News announcements |
| step_by_step_nav | 10 | Step-by-step task guides |
| hmrc_manual | 10 | HMRC internal manual cover pages |
| simple_smart_answer | 8 | Single-question tools |
| manual_section | 6 | Sections of a guidance manual |
| manual | 4 | Guidance manual cover pages |
| international_treaty | 4 | UK tax treaties |
| policy_paper | 3 | Technical notes / RCB briefs |
| local_transaction | 2 | Council-delivered services |
| finder | 2 | Search interfaces (incl. Find HMRC manuals) |
| corporate_report | 2 | Official transparency reports |
| recruitment | 1 | HMRC jobs page |
| promotional | 1 | Service promotion |
| organisation | 1 | HMRC homepage |
| news_story | 1 | General news article |
| national_statistics | 1 | Official statistics |
| correspondence | 1 | Published newsletters/letters |
| about | 1 | About HMRC page |
