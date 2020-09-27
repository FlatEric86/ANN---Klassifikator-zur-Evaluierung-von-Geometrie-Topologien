# ANN - Klassifikator-zur-Evaluierung-von-Geometrie-Topologien
Zur Projektbeschreibung lesen Sie bitte "Projektbeschreibung.pdf".

# Zusatz zur Projektbeschreibung:
Die Genauigkeit des Modells mit ~0.95 ist relativ betrachtet wenig gut, da der Hauptalgorithmus zur Erzeugung der 3D-druckbaren Struktur
zu 100% genau arbeiten muss bzw. anders formuliert, invalide Tetraedertopologien in Dateien münden, welche möglicherweise nicht 3D-druckbar sind.
Jedoch könnte ein Grund für die Ungenauigkeit von 0.05 in falsch gelabelten Trainingsdaten liegen. 
Ein anderer Grund könnten mögliche Rundungsfehler sein.
Das Problem musste damals auf andere Art gelöst werden, weshalb ich diesen Ansatz verworfen und nicht weiter bearbeitet habe. Jedoch werde ich bei 
Gelegenheit der Hypothese, dass einige Trainingsdaten falsch gelabelt wurden, nachgehen. Denn letztendlich bildet das ANN auf ähnliche weise die lineare Algebra ab, welche ich alternativ zum Lösen des Problems mit 100'iger Genauigkeit genutzt habe.
