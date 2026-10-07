"""Fictional client matters. The held-out set was written for blind-adapters before any
teacher data existed and is never sent to the teacher or used in training. The dev set
is the blind-counsel final round (its 8 held-out matters and 2 dev matters).

All names, firms and events are fictional.
"""
from __future__ import annotations

# 12 held-out matters: 4 in-scope, 6 novel in-domain, 2 out-of-scope.
HELD_OUT = {
    # ---- in-scope, like the blind-counsel set
    "IS-UD1": "Note of client meeting, 6 October 2026. Client: Priya Nair. Employer: Kestrel Analytics Ltd. "
              "Ms Nair started work as a data analyst on 3 February 2022. On 4 September 2026 she was given "
              "four weeks' notice of dismissal for alleged misconduct after a dispute about expenses, and her "
              "employment ended when that notice expired on 2 October 2026. There was no investigation and no "
              "disciplinary hearing. Her salary was £52,000 a year. She has not yet presented a claim to an "
              "employment tribunal.",
    "IS-UD2": "Note of client meeting, 28 September 2026. Client: Callum Reid. Employer: Harbour View Care Ltd. "
              "Mr Reid worked as a care home cook from 11 April 2016. He was dismissed with immediate effect on "
              "19 June 2026 after his manager said the kitchen had failed a hygiene inspection. He was not given "
              "notice or pay in lieu of notice. His salary was £27,040 a year. He has not contacted ACAS and has "
              "not presented a claim.",
    "IS-R1": "Note of client meeting, 7 October 2026. Client: Sandra Okoye, aged 47. Employer: Fenwick Components "
             "Ltd. Ms Okoye worked as a quality inspector from 1 September 2013. On 1 September 2026 she was given "
             "written notice that the company was closing its Coventry factory and that her employment would end "
             "on 30 September 2026; it ended on that date. She was not offered any other job. Her gross weekly "
             "pay was £820.",
    "IS-R2": "Note of client meeting, 2 October 2026. Client: Ben Achterberg, aged 24. Employer: Tidewell "
             "Logistics Ltd. Mr Achterberg worked as a warehouse operative from 14 November 2024. On 11 September "
             "2026 the company told him it had lost its main contract and needed fewer warehouse staff, and "
             "dismissed him for redundancy with immediate effect, without notice. He was not offered any other "
             "job. His gross weekly pay was £480.",
    # ---- novel: constructive dismissal, s95(1)(c)
    "NV-CD1": "Note of client meeting, 30 September 2026. Client: Rachel Dunmore. Employer: Ashgrove Estate Agents "
              "Ltd. Ms Dunmore started work as a lettings manager on 7 January 2021. On 1 July 2026 the managing "
              "director moved her to a junior negotiator role, took her team away from her and, at a staff "
              "meeting, accused her of dishonesty without giving any reason. Her contract does not allow the "
              "employer to change her role and she did not agree to the move. She complained in writing and "
              "received no reply. She resigned without notice on 4 September 2026 because of the demotion and "
              "the accusation. Her salary was £46,000 a year. She has not presented a claim.",
    "NV-CD2": "Note of client meeting, 25 September 2026. Client: Martin Kowalczyk. Employer: Elmfield Bakeries "
              "Ltd. Mr Kowalczyk started work as a production baker on 2 May 2019. His written contract says the "
              "employer may change his shift pattern on four weeks' written notice. On 3 August 2026 the employer "
              "gave him four weeks' written notice that he would move from day shifts to early-morning shifts, "
              "with no change in pay. He did not like the new hours and resigned on 14 August 2026, giving one "
              "week's notice; his last day was 21 August 2026. He has no other complaint about the employer. His "
              "salary was £29,500 a year. He wants to know whether he can claim unfair dismissal.",
    # ---- novel: automatically unfair reasons, no qualifying period (s108(3))
    "NV-AU1": "Note of client meeting, 18 September 2026. Client: Sofia Marchetti. Employer: Brightline Dental "
              "Ltd. Ms Marchetti started work as a dental nurse on 6 January 2026. On 3 August 2026 she told the "
              "practice manager that she was pregnant. On 24 August 2026 the practice manager dismissed her with "
              "immediate effect; the dismissal letter says that the practice cannot cover her maternity leave "
              "next year. There had been no complaints about her work. Her salary was £26,000 a year. She has not "
              "presented a claim.",
    "NV-AU2": "Note of client meeting, 1 October 2026. Client: Daniel Okonkwo. Employer: Riverside Pharmacy Group "
              "Ltd. Mr Okonkwo started work as a pharmacy technician on 1 July 2025. On 2 September 2026 he "
              "emailed the superintendent pharmacist to report that out-of-date controlled drugs were being "
              "dispensed to patients at his branch, which he believed was unlawful and put patients at risk. On "
              "9 September 2026 the area manager dismissed him with immediate effect and told him that people who "
              "go over her head do not last here. No other reason was given. His salary was £24,960 a year. He "
              "has not presented a claim.",
    # ---- novel: offer of suitable alternative employment, s141
    "NV-SA1": "Note of client meeting, 29 September 2026. Client: Gareth Pryce, aged 52. Employer: Northwind "
              "Insurance Services Ltd. Mr Pryce worked as a claims handler from 5 March 2012. On 3 August 2026 the "
              "company told him in writing that his team at the Swansea office was being cut from ten claims "
              "handlers to six and that his job would end on 28 August 2026. In the same letter it offered him a "
              "claims handler job in the motor claims team in the same Swansea office, starting on 31 August 2026, "
              "on the same pay, hours and other terms. He refused the offer in writing on 10 August 2026, saying "
              "only that he would rather take a redundancy payment. His employment ended on 28 August 2026. His "
              "gross weekly pay was £640.",
    "NV-SA2": "Note of client meeting, 5 October 2026. Client: Lucy Hamblin, aged 35. Employer: Coastline Travel "
              "Ltd. Ms Hamblin worked as a senior travel consultant at the Plymouth branch from 2 June 2019. On 1 "
              "September 2026 the company gave her written notice that the Plymouth branch would close and that "
              "her employment would end on 30 September 2026. Before her employment ended it offered her a junior "
              "booking clerk job at its Leeds office, about 300 miles away, starting on 5 October 2026, at a salary "
              "a little over half of her current salary. She has two young children at school in Plymouth. She "
              "refused the offer on 8 September 2026. Her employment ended on 30 September 2026. Her gross weekly "
              "pay was £600.",
    # ---- out of scope
    "OS-1": "Note of client meeting, 22 September 2026. Client: Tariq Hussain. Employer: Granite City Motors Ltd. "
            "Mr Hussain is still employed as a vehicle technician and has not resigned or been dismissed. Since "
            "June 2026 the employer has deducted £150 a month from his pay to cover tools it says he damaged. He "
            "never agreed to these deductions in writing. He wants to know how to get the money back.",
    "OS-2": "Note of client meeting, 30 September 2026. Client: Eleanor Shaw. Ms Shaw rented a flat in Bristol "
            "from a private landlord, Philip Grant, from 1 September 2024 to 31 August 2026 and paid a deposit of "
            "£1,450. The landlord has kept the whole deposit, saying the flat needed redecorating. She cannot "
            "find any record that the deposit was protected in a deposit scheme. She wants to know what she can "
            "claim.",
}

# The blind-counsel final round, used here as dev data (tuning allowed).
DEV = {
    "UD-A": "Note of client meeting, 9 September 2026. Client: Marcus Bell. Employer: Corvid Software Ltd. "
            "Mr Bell started work as a QA engineer on 2 March 2024. On 10 July 2026 he was given four weeks' "
            "notice of dismissal for alleged poor performance, and his employment ended when that notice "
            "expired on 7 August 2026. He had received no warnings and no performance review before the "
            "dismissal. His salary was £41,600 a year. He has not yet presented a claim to an employment tribunal.",
    "UD-B": "Note of client meeting, 15 June 2026. Client: Leah Fontaine. Employer: Brightwater Hotels Ltd. "
            "Ms Fontaine started work as a front-of-house manager on 1 June 2024. On 28 May 2026 the general "
            "manager told her in a meeting that she was dismissed with immediate effect after a guest complaint, "
            "and she was not given any notice or pay in lieu of notice. Her salary was £34,320 a year. "
            "She has not presented a claim.",
    "UD-C": "Note of client meeting, 20 April 2026. Client: Owen Hartley. Employer: Pinecrest Builders Ltd. "
            "Mr Hartley started work as a site supervisor on 5 October 2024. He was dismissed with immediate "
            "effect on 31 March 2026 after a dispute with a customer. His salary was £39,000 a year. "
            "He has not presented a claim.",
    "UD-D": "Note of client meeting, 5 May 2026. Client: Farah Siddiqui. Employer: Northgate Logistics Ltd. "
            "Ms Siddiqui worked as a transport planner from 14 January 2019. She was dismissed with immediate "
            "effect on 2 February 2026 for alleged gross misconduct. Her salary was £45,500 a year. She has not "
            "contacted ACAS and has not presented a claim.",
    "UD-E": "Note of client meeting, 1 July 2026. Client: Jonathan Price. Employer: Ardent Capital LLP. "
            "Mr Price was head of risk from 3 September 2018. He was dismissed with immediate effect on 12 June "
            "2026; the firm said his role was being restructured, but he believes this was a pretext and wants "
            "to bring an unfair dismissal claim. His salary was £182,000 a year. He has not presented a claim.",
    "R-A": "Note of client meeting, 14 August 2026. Client: Grace Mensah, aged 30. Employer: Lumen Retail Ltd. "
           "Ms Mensah worked as a store supervisor from 1 July 2020. On 3 July 2026 she was given written notice "
           "that her store would close and that her job would end on 31 July 2026; her employment ended on "
           "31 July 2026. No alternative job was offered. Her gross weekly pay was £500. She wants to know "
           "whether she is owed a statutory redundancy payment and how much.",
    "R-B": "Note of client meeting, 10 September 2026. Client: Daniel Reyes, aged 27. Employer: Quayside Print "
           "Ltd. Mr Reyes worked as a machine operator from 4 January 2025. On 7 August 2026 the company told "
           "him the print room was closing and gave him one week's notice of dismissal for redundancy, ending "
           "on 14 August 2026. His gross weekly pay was £520. He wants to know whether he is entitled to a "
           "statutory redundancy payment.",
    "OOS": "Note of client meeting, 22 September 2026. Client: Hannah Osei. Employer: Redfern Accountancy LLP. "
           "Ms Osei is still employed as a senior associate. Since she told her manager in June 2026 that she "
           "was pregnant, she has been removed from two client accounts and was passed over for a promotion "
           "that went to a less experienced colleague. She has not resigned and has not been dismissed. She "
           "wants to know what claim she might have.",
    "dismissal": "Note of client meeting, 15 May 2026. Client: Adaeze Okafor. Employer: Meridian Freight plc, "
                 "Daventry depot. Ms Okafor started work as a dispatch supervisor on 4 August 2023 under a "
                 "written contract of employment. On 12 March 2026 her manager, Dave Pritchard, telephoned her "
                 "and told her she was dismissed with immediate effect for gross misconduct, namely three days' "
                 "unauthorised absence from 16 to 18 February 2026. She says she emailed Mr Pritchard on 15 "
                 "February explaining that her daughter had been admitted to hospital, and she has kept a copy "
                 "of that email. No investigation meeting or disciplinary hearing took place and she was not "
                 "offered an appeal. She received a short dismissal letter dated 13 March 2026. Her gross pay "
                 "was £48,500 a year (about £932 a week). She has been unable to find new work. She has not yet "
                 "contacted ACAS or presented a claim to an employment tribunal.",
    "redundancy": "Note of client meeting, 2 June 2026. Client: Tomasz Wisniewski, aged 44. Employer: Halcyon "
                  "Print Ltd. He has worked as a press operator at the Leicester site since 1 March 2017. On 30 "
                  "April 2026 he received written notice that the Leicester site would close because the company "
                  "is moving all printing to its Glasgow site, and that his employment would end on 29 May 2026. "
                  "He was not offered a role in Glasgow or anywhere else. His gross weekly pay was £610. The "
                  "employer has told him he will receive a 'goodwill payment' but has said nothing about "
                  "statutory redundancy pay.",
}
