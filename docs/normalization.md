# Text normalization

Raw `business_name`, `business_address`, and `country` values are never
modified. The candidate generator derives separate normalized keys using
Unicode NFKC, `casefold`, ampersand-to-`and`, punctuation-as-separator, and
collapsed whitespace.

Names additionally discard a conservative set of trailing legal-form tokens
(for example `pvt ltd`, `private limited`, `inc`, `sarl`) only while at least
one other token remains. Legal words in the middle of a name stay intact.
Addresses receive only generic text normalization, retaining house/building
numbers, postal codes, unit-like alphanumeric identifiers, and non-Latin text.
No US- or India-specific address assumptions are made; France and future
countries use the same deterministic logic.

The current policy intentionally does not ASCII-fold accents because generic
combining-mark removal can damage Indic text. This trade-off will be evaluated
in blocking recall rather than assumed beneficial.
