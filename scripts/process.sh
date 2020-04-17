export CLASSPATH=$(realpath ../stanford-corenlp/stanford-corenlp-3.9.2.jar)
echo "Please tokenize this text." | java edu.stanford.nlp.process.PTBTokenizer

if [ $# -eq 3 ]; then
    dataDir=$(realpath $1)

    if [ -d $dataDir ] && [ -d "$dataDir/articles" ] && [ -d "$dataDir/summaries" ]; then
        python ../pg-datatool/make_stories.py "$dataDir/articles" "$dataDir/summaries" &&
        outDir=$(realpath $2); trainRate=$3
        python data/make_datafiles.py "$dataDir/stories" $outDir $trainRate
    else
        echo "Missing required one or more directories, check: "
        echo "$dataDir, or"
        echo "$dataDir/articles, or"
        echo "$dataDir/summaries"

    fi
else
    echo "$# out of 3 arguments provided"
    echo "Syntax: process.sh storiesDir outputDir trainRate"
fi
