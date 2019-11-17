#[cfg(test)]
mod tests {
    use crate::cluster;

    const LINES: [&str; 6] = [
        "流行,流行,流行,流行,流行,流行,流行,流行,流行,流行,蓝调,蓝调,蓝调,蓝调,蓝调,蓝调,摇滚,摇滚,摇滚,摇滚",
        "爵士,爵士,爵士,爵士,爵士,爵士,爵士,爵士,舞曲,舞曲,舞曲,舞曲,舞曲,舞曲,舞曲,舞曲,舞曲",
        "古典,古典,古典,古典,民谣,民谣,民谣,民谣",
        "爵士,爵士,爵士,爵士,爵士,爵士,爵士,爵士,爵士,金属,金属,舞曲,舞曲,舞曲,舞曲,舞曲,舞曲",
        "流行,流行,流行,流行,摇滚,摇滚,摇滚,嘻哈,嘻哈,嘻哈",
        "古典,古典,古典,古典,古典,古典,古典,古典,摇滚"];

    #[test]
    fn it_works() {
        let zhao_yi: Vec<_> = LINES[0].split(',').collect();
        let qian_er: Vec<_> = LINES[1].split(',').collect();
        let zhang_san: Vec<_> = LINES[2].split(',').collect();
        let li_si: Vec<_> = LINES[3].split(',').collect();
        let wang_wu: Vec<_> = LINES[4].split(',').collect();
        let ma_liu: Vec<_> = LINES[5].split(',').collect();

        let mut ca = cluster::ClusterAnalyzer::new();
        ca.add_doc("马六", ma_liu);
        ca.add_doc("王五", wang_wu);
        ca.add_doc("李四", li_si);
        ca.add_doc("张三", zhang_san);
        ca.add_doc("钱二", qian_er);
        ca.add_doc("赵一", zhao_yi);
        let result = ca.kmeans(3);
        for r in result {
            println!("{:?}", r);
        }
    }

    use jieba_rs;
    #[test]
    fn example() {
        let jieba = jieba_rs::Jieba::new();
        let docs: Vec<(String, Vec<&str>)> = LINES[..]
            .iter()
            .enumerate()
            .map(|(i, line)| {
                let mut tags = jieba.tag(line, false);
                tags.retain(|t| !t.tag.starts_with('x'));
                let words: Vec<_> = tags.iter().map(|t| t.word).collect();
                (format!("Mr. {}", i + 1), words)
            })
            .collect();
        let mut ca = cluster::ClusterAnalyzer::new();
        for (name, words) in docs {
            ca.add_doc(name, words);
        }
        for set in ca.kmeans(3) {
            println!("{:?}", set);
        }
    }
}

pub mod cluster;

pub use jieba_rs;
