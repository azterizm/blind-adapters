# IS-UD1, arm AT: uk_unfair_dismissal
Status: ANSWERED (every step computed or grounded)

## dismissed: YES [grounded]
Under section 95(1)(a), an employee is dismissed if the contract under which they are employed is terminated by the employer, whether with or without notice. Kestrel Analytics Ltd gave Ms Nair four weeks' notice of dismissal, which expired on 2 October 2026, terminating her employment. Under section 97(1)(a), the effective date of termination is the date on which that notice expired.
  > "On 4 September 2026 she was given four weeks' notice of dismissal for alleged misconduct after a dispute about expenses, and her employment ended when that notice expired on 2 October 2026."
  authority: uk/ukpga/1996/18/s95, uk/ukpga/1996/18/s97

## qualifying_service: YES [computed]
Continuous employment from 2022-02-03 to the effective date of termination 2026-10-02: 4 complete year(s) against the 2-year requirement in s108(1).
  > "she was given four weeks' notice of dismissal"
  > "On 4 September 2026 she was given four weeks' notice of dismissal"
  > "Ms Nair started work as a data analyst on 3 February 2022."
  > "her employment ended when that notice expired on 2 October 2026."
  authority: uk/ukpga/1996/18/s108, uk/ukpga/1996/18/s97, uk/ukpga/1996/18/s86

## automatic_unfair_reason: NO [grounded]
The dismissal was for alleged misconduct arising from a dispute about expenses, and there is no evidence that the dismissal was related to family or parental leave under section 99, health and safety matters under section 100, or a protected disclosure under section 103A. Consequently, none of the statutory exceptions in section 108(3) apply to disapply the two-year continuous employment requirement.
  > "On 4 September 2026 she was given four weeks' notice of dismissal for alleged misconduct after a dispute about expenses, and her employment ended when that notice expired on 2 October 2026."
  authority: uk/ukpga/1996/18/s100, uk/ukpga/1996/18/s103A, uk/ukpga/1996/18/s108, uk/ukpga/1996/18/s99

## time_limit: YES (2027-01-01) [computed]
Primary limit under s111(2)(a): 3 months beginning with the EDT 2026-10-02, i.e. on or before 2027-01-01; no claim presented yet; 87 day(s) remain at the advice date 2026-10-06. (ACAS early-conciliation extensions under s207B and the 'not reasonably practicable' extension are not computed.)
  > "Note of client meeting, 6 October 2026"
  > "her employment ended when that notice expired on 2 October 2026."
  authority: uk/ukpga/1996/18/s111, uk/ukpga/1996/18/s97

## fairness: YES [grounded]
Under section 98(1), the employer must establish the reason for the dismissal and show that it falls within a statutory category or is some other substantial reason. Dismissing an employee for alleged misconduct without an investigation or disciplinary hearing fails the test of reasonableness under section 98(4).
  > "On 4 September 2026 she was given four weeks' notice of dismissal for alleged misconduct after a dispute about expenses, and her employment ended when that notice expired on 2 October 2026."
  > "There was no investigation and no disciplinary hearing."
  authority: uk/ukpga/1996/18/s98

## compensation_cap: YES (52000) [computed]
s124(1ZA): the lower of £123,543 and 52 x a week's pay (£1,000.00) = £52,000. This caps the compensatory award; it does not decide whether there is a loss.
  > "Her salary was £52,000 a year."
  authority: uk/ukpga/1996/18/s123, uk/ukpga/1996/18/s124
