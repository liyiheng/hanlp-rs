use jieba_rs;
use mining::cluster;
use std::io::Read;
use std::time;

fn main() {
    println!("hello");
    let mut file = std::fs::File::open("./intros.dat").unwrap();
    let mut all = String::new();
    file.read_to_string(&mut all).unwrap();
    let start = time::Instant::now();
    let jieba = jieba_rs::Jieba::new();
    let lines: Vec<(&str, Vec<&str>)> = all
        .split('\n')
        .filter(|l| l.len() > 1)
        .map(|line| {
            let parts: Vec<&str> = line.splitn(2, ' ').collect();
            let mut words = jieba.tag(parts[1], false);
            words.retain(|t| !t.tag.starts_with('x'));
            let words = words.into_iter().map(|t| t.word).collect();
            (parts[0], words)
        })
        .collect();
    println!("cut words done {}ms", start.elapsed().as_millis());
    let mut ca = cluster::ClusterAnalyzer::new();
    for (user, words) in lines.into_iter() {
        ca.add_doc(user, words);
    }
    println!("add docs done {}ms", start.elapsed().as_millis());
    let mut result = ca.kmeans(1000);
    println!("kmeans done {}ms", start.elapsed().as_millis());
    result.sort_by(|a, b| a.len().cmp(&b.len()));
    println!("sort done {}ms", start.elapsed().as_millis());
    for set in result {
        let ids: Vec<_> = set.into_iter().collect();
        println!("{:?}", ids);
    }
    println!("total: {}ms", start.elapsed().as_millis());
}
