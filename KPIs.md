# Key KPIs: Understanding How Verizon's Business is Measured

This document provides a detailed breakdown of Verizon's KPIs, their definitions, formulas, and source tables.

---

## Gross Adds
- **Definition**: Comprises three activity codes:
  - **Activations (ACTIVITY_CD = 'AC')**: Represents a new line of service.
  - **30-Day Deactivations (ACTIVITY_CD = 'D3')** and **60-Day Deactivations (ACTIVITY_CD = 'D6')**: Lines deactivated within 30 or 60 days.
- **Formula**:
```
Gross Adds = ACTIVITY_CD = 'AC' - (ACTIVITY_CD = 'D3' or ACTIVITY_CD = 'D6')
```
- **Key Note**: D3 and D6 offset AC transactions to ensure 30/60-day disconnects are excluded.
- **Source Table**: `DLA_SUM_FACT_V`

---

## Gross Add Categories
1. **Add a Line**:
 - A new line added to an existing account after 60 days of account establishment.
 - **Source Table**: `DLA_SUM_FACT_V`
2. **New to Verizon**:
 - Activations occurring within 60 days of account establishment.
 - **Source Table**: `DLA_SUM_FACT_V`

---

## Disconnects
1. **Voluntary Disconnects**:
 - Customers leaving for reasons other than non-pay.
 - Offset by voluntary reactivations (voided if returning within 30 days).
 - **Formula**:
   ```
   Net Disconnect = VOLUNTARY_DISCONNECTS - VOLUNTARY_REACTIVATIONS
   ```
 - **Source Table**: `DLA_SUM_FACT_V`
2. **Involuntary Disconnects**:
 - Customers leaving due to non-pay reasons.
 - Offset by involuntary reactivations (voided if returning within 30 days).
 - **Formula**:
   ```
   Net Disconnect = INVOLUNTARY_DISCONNECTS - INVOLUNTARY_REACTIVATIONS
   ```
 - **Source Table**: `DLA_SUM_FACT_V`

---

## Net Adds
- **Definition**: Total activations minus deactivations.
- **Formula**:
Net Adds = GROSS_ADDS - VOLUNTARY_DISCONNECTS - INVOLUNTARY_DISCONNECTS

- **Source Table**: `DLA_SUM_FACT_V`

---

## Churn
- **Definition**: Measures customer attrition.
- **Formula**:
Churn = DISCONNECTS / ((BOP_SUBS + EOP_SUBS) / 2)

- **Key Note**: BOP_SUBS (beginning of period subscribers) and EOP_SUBS (end of period subscribers) are required for this calculation.
- **Source Tables**: `DLA_SUM_FACT_V`, `SUBS_SUM_FACT_V`

---

## Sales Metrics
1. **Sales Quantity**:
 - Represents device or accessory sales.
 - **Source Table**: `EQUIP_SUM_FACT_V`
2. **Return Quantity**:
 - Represents product returns.
 - **Source Table**: `EQUIP_SUM_FACT_V`
3. **Net Sales**:
 - **Formula**:
   ```
   Net Sales = Total Sales - Returns
   ```
 - **Source Table**: `EQUIP_SUM_FACT_V`

---

## Acquisitions and Upgrades
1. **New Acquisition**:
 - First device ever purchased by a customer.
 - **Source Table**: `EQUIP_SUM_FACT_V`
2. **Upgrades**:
 - Additional devices purchased by existing customers.
 - **Formula**:
   ```
   Upgrades = FIN_TOT_FLAG = 'Y' AND ACQ_RET_IND = 'R'
   ```
 - **Source Table**: `EQUIP_SUM_FACT_V`

---

## Margin Metrics
1. **Equipment Margin**:
 - **Formula**:
   ```
   Equipment Margin = ITEM_PRICE - ITEM_COST - Discounts - Rebates
   ```
 - **Source Table**: `EQUIP_SUM_FACT_V`
2. **Accessory Margin**:
 - **Formula**:
   ```
   Accessory Margin = (Quantity * Price) - Cost - Discounts
   ```
 - **Source Table**: `EQUIP_SUM_FACT_V`

---

## Subscriber Metrics
1. **Ending Customers**:
 - Total active customers at the end of a reporting period.
 - **Source Table**: `SUBS_SUM_FACT_V`
2. **Billed Lines**:
 - Lines billed during a reporting month.
 - **Source Table**: `REV_SUM_FACT_BL_V`

---

## Revenue Metrics
1. **Total Service Revenue**:
 - Includes access charges, overages, roaming, and device protection.
 - **Source Table**: `REV_SUM_FACT_BL_V`
2. **Average Revenue Per User (ARPU)**:
 - **Formula**:
   ```
   ARPU = Total Service Revenue / Billed Lines
   ```
 - **Source Table**: `REV_SUM_FACT_BL_V`
3. **Average Revenue Per Account (ARPA)**:
 - **Formula**:
   ```
   ARPA = Total Service Revenue / Billed Account Fractions
   ```
 - **Source Table**: `REV_SUM_FACT_BL_V`

---

## Usage Metrics
- **Definition**: Tracks data usage, minutes of use, SMS, and MMS.
- **Source Table**: `REV_SUM_FACT_BL_V`

---

## Common Filters
- **Reporting Month**: (`RPT_MTH`)
- **Prepaid Indicator**: (`PREPAID_IND`)
- **Line Type Code**: (`LINE_TYPE_CD`)
- **Revenue Generating Indicator**: (`REV_GEN_IND`)
- **Managed Market Indicator**: (`MANAGED_IND`)

---

## Common Dimensions
- **Reporting Month** (`RPT_MTH`)
- **Device Grouping** (`EQP_GRP_DESC`)
- **Price Plan Type** (`COE_PPLAN_TYPE_DESC`)
- **Geographical Region** (`REGION_DESC`)
- **Customer Segment** (`VZ2_SEGMT_CTGRY_DESC`)

---

## Common Join Columns
- **Verizon Segment Code** (`VZ2_SEGMT_CD`)
- **Price Plan Code** (`PPLAN_CD`)
- **Sales Outlet ID** (`SLS_OUTLET_ID`)
- **Equipment Product Name** (`EQP_PROD_NM`)
- **Market Code** (`MKT_CD`)
