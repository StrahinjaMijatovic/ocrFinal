import json
import pyodbc
from app.config import settings

SQL_QUERY = """
SELECT (
    SELECT
        vr.Id                                               AS zahtev_id,
        vr.Code                                             AS zahtev_kod,
        CONVERT(VARCHAR, vr.SubmissionDate, 23)             AS datum_podnosenja,
        CONVERT(VARCHAR, vr.CreatedDate, 23)                AS datum_kreiranja,
        vr.SubmittedBy                                      AS podneo,
        vr.IsMinor                                          AS je_maloletnik,
        vr.IsFromEXPO                                       AS expo_zahtev,
        vr.IsDigital                                        AS digitalni_zahtev,
        vr.Priority                                         AS trenutni_prioritet,
        vr.ConsulOpinion                                    AS misljenje_konzula,
        vs.Name                                             AS status,
        vs.NameEnglish                                      AS status_en,
        vc.Name                                             AS kategorija_vize,
        vc.NameEnglish                                      AS kategorija_vize_en,
        vc.Code                                             AS kategorija_kod,
        vd.NumberOfDays                                     AS broj_dana_boravka,
        CONVERT(VARCHAR, vd.ArrivalDate, 23)                AS datum_dolaska,
        CONVERT(VARCHAR, vd.DepartureDate, 23)              AS datum_odlaska,
        vd.TransportMeans                                   AS prevozno_sredstvo,
        vd.IsPersonInRS                                     AS vec_u_srbiji,
        vd.PreviousStaysInRs                                AS prethodni_boravci,
        vd.OtherVisas                                       AS druge_vize,
        tp.Name                                             AS svrha_putovanja,
        tp.NameEnglish                                      AS svrha_putovanja_en,
        stp.Name                                            AS podsvrha_putovanja,
        stp.NameEnglish                                     AS podsvrha_putovanja_en,
        noe.Name                                            AS broj_ulazaka,
        noe.NameEnglish                                     AS broj_ulazaka_en,
        bc.Name                                             AS granicni_prelaz,
        bc.NameEnglish                                      AS granicni_prelaz_en,
        vra.FirstName                                       AS ime,
        vra.LastName                                        AS prezime,
        vra.BirthLastName                                   AS rodjeno_prezime,
        CONVERT(VARCHAR, vra.BirthDate, 23)                 AS datum_rodjenja,
        vra.BirthPlace                                      AS mesto_rodjenja,
        vra.PersonalIdNumber                                AS jmbg,
        vra.Phone                                           AS telefon,
        vra.Email                                           AS email,
        vra.FathersName                                     AS ime_oca,
        vra.MothersName                                     AS ime_majke,
        vra.Address                                         AS adresa,
        vra.PlaceOfResidence                                AS mesto_boravka,
        g.Name                                              AS pol,
        c.Name                                              AS drzavljanstvo,
        c.NameEnglish                                       AS drzavljanstvo_en,
        c.Code                                              AS drzavljanstvo_kod,
        td.DocumentNumber                                   AS broj_dokumenta,
        td.IssuedBy                                         AS izdao_dokument,
        CONVERT(VARCHAR, td.IssueDate, 23)                  AS datum_izdavanja_dokumenta,
        CONVERT(VARCHAR, td.ExpiryDate, 23)                 AS datum_isteka_dokumenta,
        td.PermissionDocumentForReturn                      AS ima_dozvolu_povratka,
        DATEDIFF(DAY, GETDATE(), td.ExpiryDate)             AS pasos_istice_za_dana,
        hd.HostName                                         AS domacin_ime,
        hd.HostTelephone                                    AS domacin_telefon,
        hd.HostEmail                                        AS domacin_email,
        hd.HostAddress                                      AS domacin_adresa,
        hd.MeansOfSupportDescription                        AS nacin_izdrzavanja,
        inv.FullName                                        AS pozivalac_ime,
        inv.Phone                                           AS pozivalac_telefon,
        inv.Email                                           AS pozivalac_email,
        inv.LegalEntityName                                 AS pozivalac_firma,
        inv.RegistrationNumber                              AS pozivalac_maticni_br,
        it.Name                                             AS tip_pozivara,
        it.NameEnglish                                      AS tip_pozivara_en,
        (SELECT COUNT(*) FROM dbo.DocumentUploads du WHERE du.VisaRequestId = vr.Id) AS broj_uploadovanih_dokumenata,
        (SELECT COUNT(*) FROM dbo.VisaRequests vr2
         JOIN dbo.VisaRequestApplicants vra2 ON vra2.VisaRequestId = vr2.Id AND vra2.PeriodEnd > GETDATE()
         WHERE vra2.PersonalIdNumber = vra.PersonalIdNumber AND vr2.Id != vr.Id AND vr2.PeriodEnd > GETDATE()
        )                                                   AS broj_prethodnih_zahteva,
        DATEDIFF(DAY, GETDATE(), vd.ArrivalDate)            AS dana_do_dolaska,
        DATEDIFF(DAY, vd.ArrivalDate, vd.DepartureDate)     AS duzina_boravka_dana,
        DATEDIFF(YEAR, vra.BirthDate, GETDATE())            AS starost,
        CASE
            WHEN DATEDIFF(DAY, GETDATE(), vd.ArrivalDate) < 3  THEN 'URGENCY_CRITICAL'
            WHEN DATEDIFF(DAY, GETDATE(), vd.ArrivalDate) < 8  THEN 'URGENCY_HIGH'
            WHEN DATEDIFF(DAY, GETDATE(), vd.ArrivalDate) < 15 THEN 'URGENCY_MEDIUM'
            ELSE 'URGENCY_LOW'
        END                                                 AS urgency_kategorija,
        CASE
            WHEN vr.IsMinor = 1                                    THEN 'APPLICANT_MINOR'
            WHEN DATEDIFF(YEAR, vra.BirthDate, GETDATE()) > 70    THEN 'APPLICANT_ELDERLY'
            ELSE 'APPLICANT_STANDARD'
        END                                                 AS starosna_kategorija,
        CASE
            WHEN DATEDIFF(DAY, GETDATE(), td.ExpiryDate) < 0   THEN 'PASSPORT_EXPIRED'
            WHEN DATEDIFF(DAY, GETDATE(), td.ExpiryDate) < 180 THEN 'RISK_PASSPORT_EXPIRING'
            ELSE 'PASSPORT_OK'
        END                                                 AS pasos_status
    FROM dbo.VisaRequests vr
    LEFT JOIN dbo.VisaStatuses          vs  ON vs.Id  = vr.VisaStatusId
    LEFT JOIN dbo.VisaData              vd  ON vd.VisaRequestId = vr.Id AND vd.PeriodEnd > GETDATE()
    LEFT JOIN dbo.VisaCategories        vc  ON vc.Id  = vd.VisaCategoryId
    LEFT JOIN dbo.SubTripPurposes       stp ON stp.Id = vd.SubTripPurposeId
    LEFT JOIN dbo.TripPurposes          tp  ON tp.Id  = stp.TripPurposeId
    LEFT JOIN dbo.NumberOfEntries       noe ON noe.Id = vd.NumberOfEntryId
    LEFT JOIN dbo.BorderCrossings       bc  ON bc.Id  = vd.BorderCrossingId
    LEFT JOIN dbo.VisaRequestApplicants vra ON vra.VisaRequestId = vr.Id AND vra.PeriodEnd > GETDATE()
    LEFT JOIN dbo.Genders               g   ON g.Id   = vra.GenderId
    LEFT JOIN dbo.Citizenships          c   ON c.Id   = vra.NationalityId
    LEFT JOIN dbo.TravelDocuments       td  ON td.VisaRequestId = vr.Id
    LEFT JOIN dbo.HostData              hd  ON hd.VisaRequestId = vr.Id AND hd.PeriodEnd > GETDATE()
    LEFT JOIN dbo.Inviters              inv ON inv.VisaRequestId = vr.Id AND inv.PeriodEnd > GETDATE()
    LEFT JOIN dbo.InviterTypes          it  ON it.Id  = inv.InviterTypeId
    WHERE vr.Id = ? AND vr.PeriodEnd > GETDATE()
    FOR JSON PATH, WITHOUT_ARRAY_WRAPPER
) AS rezultat
"""


def get_db_connection():
    conn_str = (
        f"DRIVER={{{settings.db_driver}}};"
        f"SERVER={settings.db_server};"
        f"DATABASE={settings.db_name};"
        "Trusted_Connection=yes;"
    )
    return pyodbc.connect(conn_str)


def fetch_request_data(request_id: int) -> dict:
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(SQL_QUERY, (request_id,))
        row = cursor.fetchone()

    if not row or not row[0]:
        raise ValueError(f"No data found for RequestId {request_id}")

    return json.loads(row[0])
