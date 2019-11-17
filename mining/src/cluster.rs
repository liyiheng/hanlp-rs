#![allow(dead_code)]
use rand::prelude::*;
use rayon::prelude::*;
use std::collections::BTreeMap;
use std::collections::HashMap;
use std::collections::HashSet;
use std::hash::Hash;
use std::hash::Hasher;
use std::sync::Arc;

pub struct Document<K>
where
    K: Hash + Send + Sync,
{
    pub id: K,
    pub feature: SparseVector,
}

impl<K: Hash + PartialEq + Send + Sync> Document<K> {
    pub fn new(k: K, feature: SparseVector) -> Self {
        Document { id: k, feature }
    }
}

impl<K: Hash + PartialEq + Send + Sync> PartialEq for Document<K> {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl<K: Hash + PartialEq + Send + Sync> Eq for Document<K> {}

impl<K: Hash + PartialEq + Send + Sync> Hash for Document<K> {
    fn hash<H: Hasher>(&self, h: &mut H) {
        self.id.hash(h);
    }
}

#[derive(Default)]
pub struct SparseVector(BTreeMap<i32, f64>);

impl SparseVector {
    pub fn get(&self, key: i32) -> f64 {
        *self.0.get(&key).unwrap_or(&0.0)
    }

    fn norm(&self) -> f64 {
        self.norm_squared().sqrt()
    }

    fn norm_squared(&self) -> f64 {
        self.0.values().fold(0.0, |acc, v| acc + v * v)
    }

    fn normalize(&mut self) {
        let nrm = self.norm();
        self.0.values_mut().for_each(|v| *v /= nrm);
    }

    fn multiply_const(&mut self, x: f64) {
        self.0.values_mut().for_each(|v| *v *= x);
    }

    pub fn add_vector(&mut self, v: &Self) {
        for (k, v) in v.0.iter() {
            let prev = *self.0.get(k).unwrap_or(&0.0);
            self.0.insert(*k, prev + v);
        }
    }

    pub fn sub_vector(&mut self, v: &Self) {
        for (k, v) in v.0.iter() {
            let prev = *self.0.get(k).unwrap_or(&0.0);
            self.0.insert(*k, prev - v);
        }
    }

    fn inner_product(a: &Self, b: &Self) -> f64 {
        let mut prod = 0.0;
        for (k, v) in a.0.iter() {
            prod += v * b.0.get(k).unwrap_or(&0.0);
        }
        prod
    }

    fn cosine(a: &Self, b: &Self) -> f64 {
        let norm_a = a.norm();
        let norm_b = b.norm();
        if norm_a == 0.0 && norm_b == 0.0 {
            return 0.0;
        }
        let prod = Self::inner_product(a, b);
        let result = prod / (norm_a * norm_b);
        if result.is_nan() {
            0.0
        } else {
            result
        }
    }
}

pub struct Cluster<K: Hash + PartialEq + Send + Sync> {
    documents: Vec<Option<Arc<Document<K>>>>,
    composite: SparseVector,
    centroid: SparseVector,
    sectioned_clusters: Vec<Cluster<K>>,
    sectioned_gain: f64,
}

impl<K: Hash + PartialEq + Send + Sync> PartialEq for Cluster<K> {
    fn eq(&self, other: &Self) -> bool {
        self.sectioned_gain == other.sectioned_gain
    }
}

impl<K: Hash + PartialEq + Send + Sync> PartialOrd for Cluster<K> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.sectioned_gain.partial_cmp(&other.sectioned_gain)
    }
}

impl<K: Hash + PartialEq + Send + Sync> Cluster<K> {
    fn new(docs: Vec<Option<Arc<Document<K>>>>) -> Self {
        Cluster {
            documents: docs,
            composite: SparseVector::default(),
            centroid: SparseVector::default(),
            sectioned_clusters: vec![],
            sectioned_gain: 0.0,
        }
    }

    fn clear(&mut self) {
        self.documents.clear();
        self.composite.0.clear();
        self.centroid.0.clear();
        self.sectioned_clusters.clear();
    }

    fn len(&self) -> usize {
        self.documents.len()
    }

    fn centroid_vec(&'_ mut self) -> &'_ SparseVector {
        self.centroid.0 = self.composite.0.clone();
        self.centroid.normalize();
        &self.centroid
    }

    fn remove_doc(&mut self, index: usize) {
        let doc = self.documents.get_mut(index);
        if let Some(d) = doc {
            if let Some(v) = d {
                self.composite.sub_vector(&v.feature);
            }
            *d = None;
        }
    }
    fn add_doc(&mut self, doc: Arc<Document<K>>) {
        self.composite.add_vector(&doc.feature);
        self.documents.push(Some(doc));
    }
    fn refresh(&mut self) {
        self.documents.retain(|d| d.is_some());
    }

    fn cal_sectioned_gain(&mut self) {
        let need_cal = self.sectioned_gain == 0.0 && self.sectioned_clusters.len() > 1;
        if !need_cal {
            self.sectioned_gain = 0.0;
            return;
        }
        self.sectioned_gain = self
            .sectioned_clusters
            .iter_mut()
            .fold(0.0, |acc, c| acc + c.composite.norm())
            - self.composite.norm();
    }

    fn choose_smartly(&self, mut n: usize) -> Vec<Arc<Document<K>>> {
        let mut result = vec![];
        let size = self.documents.len();
        let mut closest = vec![0.0; size];
        n = if size < n { size } else { n };
        let index = rand::thread_rng().gen_range(0, size);
        let doc = self.documents.get(index).unwrap().as_ref().unwrap().clone();
        result.push(doc.clone());
        let mut potential = 0.0;
        for (i, ele) in self.documents.iter().enumerate() {
            if let Some(ele) = ele {
                let dist = 1.0 - SparseVector::inner_product(&(*ele).feature, &(*doc).feature);
                potential += dist;
                closest[i] = dist;
            }
        }
        while result.len() < n {
            let mut r_val: f64 = rand::random::<f64>() * potential;
            let mut index = 0;
            for (i, dist) in closest.iter().enumerate() {
                index = i;
                if r_val <= *dist {
                    break;
                }
                r_val -= dist;
            }
            let cur_doc = self.documents.get(index).unwrap().as_ref().unwrap();
            result.push(cur_doc.clone());

            let mut new_potential = 0.0;
            for (i, min) in closest.iter_mut().enumerate() {
                let doc_i = self.documents.get(i).unwrap().as_ref().unwrap();
                let dist =
                    1.0 - SparseVector::inner_product(&(*doc_i).feature, &(*cur_doc).feature);
                if dist < *min {
                    *min = dist;
                }
                new_potential += *min;
            }
            potential = new_potential;
        }
        result
    }

    fn section(&mut self, n: usize) {
        if self.documents.len() < n {
            return;
        }
        let centroids = self.choose_smartly(n);
        let mut sectioned: Vec<Cluster<K>> = Vec::with_capacity(n);
        for _ in 0..centroids.len() {
            sectioned.push(Cluster::new(vec![]));
        }
        for d in self.documents.iter() {
            if d.is_none() {
                continue;
            }
            let d = d.as_ref().unwrap();
            let mut max_similarity = -1.0;
            let mut max_index = 0;
            for (i, c) in centroids.iter().enumerate() {
                let similarity = SparseVector::inner_product(&(*d).feature, &(*c).feature);
                if similarity > max_similarity {
                    max_similarity = similarity;
                    max_index = i;
                }
            }
            sectioned.get_mut(max_index).unwrap().add_doc(d.clone());
        }
        self.sectioned_clusters = sectioned;
    }
}

pub struct ClusterAnalyzer<K, W>
where
    W: Hash + Clone + Eq,
    K: Hash + Clone + Eq + Send + Sync,
{
    vocabulary: HashMap<W, i32>,
    documents: HashMap<K, Arc<Document<K>>>,
}

impl<K, W> ClusterAnalyzer<K, W>
where
    W: Hash + Clone + Eq,
    K: Hash + Clone + Eq + Send + Sync,
{
    pub fn new() -> Self {
        ClusterAnalyzer {
            vocabulary: HashMap::default(),
            documents: HashMap::default(),
        }
    }
    fn get_id(&mut self, word: &W) -> i32 {
        if self.vocabulary.contains_key(word) {
            *self.vocabulary.get(word).unwrap()
        } else {
            let l = self.vocabulary.len() as i32;
            self.vocabulary.insert(word.clone(), l);
            l
        }
    }
    pub fn add_doc(&mut self, k: K, words: Vec<W>) {
        let mut sv = SparseVector::default();
        for w in words {
            let id = self.get_id(&w);
            let cnt = *sv.0.get(&id).unwrap_or(&0.0);
            sv.0.insert(id, cnt + 1.0);
        }
        sv.normalize();
        let doc = Document::new(k.clone(), sv);
        self.documents.insert(k, Arc::new(doc));
    }

    pub fn kmeans(&self, n: usize) -> Vec<HashSet<K>> {
        let mut cluster = Cluster::new(vec![]);
        for doc in self.documents.values() {
            cluster.add_doc(doc.clone());
        }

        cluster.section(n);
        self.refine_clusters(&mut cluster.sectioned_clusters);
        cluster
            .sectioned_clusters
            .iter()
            .map(|c| {
                c.documents.iter().fold(HashSet::new(), |mut set, doc| {
                    set.insert(doc.as_ref().unwrap().id.clone());
                    set
                })
            })
            .collect()
    }

    const REFINE_LOOP_LIMIT: usize = 30;

    fn refine_clusters(&self, clusters: &mut Vec<Cluster<K>>) -> f64 {
        let mut norms = vec![0.0; clusters.len()];
        for (i, c) in clusters.iter().enumerate() {
            norms[i] = c.composite.norm();
        }
        let mut eval_cluster = 0.0;
        for _ in 0..Self::REFINE_LOOP_LIMIT {
            let mut items = Vec::with_capacity(self.documents.len());
            for (i, c) in clusters.iter().enumerate() {
                for j in 0..c.documents.len() {
                    items.push((i, j));
                }
            }
            items.shuffle(&mut rand::thread_rng());
            let mut changed = false;
            for (cluster_id, item_id) in items.iter() {
                let cluster = clusters.get(*cluster_id).unwrap();
                let doc = cluster.documents.get(*item_id).unwrap().as_ref().unwrap();
                let value_base = Self::refined_vector(&cluster.composite, &doc.feature, -1);
                let mut norm_base_moved = norms[*cluster_id].powi(2) + value_base;
                norm_base_moved = if norm_base_moved > 0.0 {
                    norm_base_moved.sqrt()
                } else {
                    0.0
                };
                let (eval_max, norm_max, index_max) = clusters
                    .par_iter()
                    .enumerate()
                    .fold(
                        || (-1.0, 0.0, 0),
                        |(v, n, i), (j, other)| {
                            if j == *cluster_id {
                                return (n, v, i);
                            }
                            let value_target =
                                Self::refined_vector(&other.composite, &doc.feature, 1);
                            let mut norm_target_moved = norms[j].powi(2) + value_target;
                            norm_target_moved = if norm_target_moved > 0.0 {
                                norm_target_moved.sqrt()
                            } else {
                                0.0
                            };
                            let eval_moved =
                                norm_base_moved + norm_target_moved - norms[*cluster_id] - norms[j];
                            if v < eval_moved {
                                (eval_moved, norm_target_moved, j)
                            } else {
                                (n, v, i)
                            }
                        },
                    )
                    .reduce(
                        || (-1.0, 0.0, 0),
                        |(n1, v1, i1), (n2, v2, i2)| {
                            if n1 > n2 {
                                (n1, v1, i1)
                            } else {
                                (n2, v2, i2)
                            }
                        },
                    );

                std::mem::drop(cluster);
                if eval_max > 0.0 {
                    changed = true;
                    norms[*cluster_id] = norm_base_moved;
                    norms[index_max] = norm_max;
                    let doc = doc.clone();
                    eval_cluster += eval_max;
                    clusters.get_mut(index_max).unwrap().add_doc(doc);
                    clusters.get_mut(*cluster_id).unwrap().remove_doc(*item_id);
                }
            }
            if !changed {
                break;
            }
            clusters.iter_mut().for_each(|c| c.refresh());
        }
        eval_cluster
    }

    // (x1, y1), (x2, y2)
    // (x1^2 - 2 x1 * x2)  + (y1^2 - 2 y1*y2) + ...
    fn refined_vector(composite: &SparseVector, vector: &SparseVector, sign: i32) -> f64 {
        if sign < 0 {
            vector.0.iter().fold(0.0, |acc, (k, v)| {
                acc + (*v).powi(2) - 2.0 * composite.get(*k) * *v
            })
        } else {
            vector.0.iter().fold(0.0, |acc, (k, v)| {
                acc + (*v).powi(2) + 2.0 * composite.get(*k) * *v
            })
        }
    }
}
