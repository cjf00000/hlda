//
// Created by jianfei on 16-10-19.
//

#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include "Tree.h"
#include "Document.h"

TEST(Tree, pos) {
    Tree tree(3, std::vector<double>{0.5, 0.5, 0.5});

    auto *root = tree.GetRoot();
    auto *node1 = tree.AddChildren(root);
    auto *node2 = tree.AddChildren(node1);

    auto *node3 = tree.AddChildren(node1);

    Document doc1, doc2;
    doc1.c = decltype(doc1.c){root, node1, node2};
    doc2.c = decltype(doc2.c){root, node1, node3};

    tree.UpdateNumDocs(doc1.c.back(), 1);
    tree.UpdateNumDocs(doc2.c.back(), 1);

    // Layer size
    EXPECT_EQ(tree.NumNodes(0), 1);
    EXPECT_EQ(tree.NumNodes(1), 1);
    EXPECT_EQ(tree.NumNodes(2), 2);

    // ID
    EXPECT_EQ(root->pos, 0);
    EXPECT_EQ(node1->pos, 0);
    EXPECT_EQ(node2->pos, 0);
    EXPECT_EQ(node3->pos, 1);

    // DocID
    ASSERT_THAT(doc1.GetPos(), testing::ElementsAre(0, 0, 0));
    ASSERT_THAT(doc2.GetPos(), testing::ElementsAre(0, 0, 1));

    tree.UpdateNumDocs(doc1.c.back(), -1);
    auto result = tree.Compress(2);

    ASSERT_THAT(result, testing::ElementsAre(-1, 0));

    EXPECT_EQ(root->pos, 0);
    EXPECT_EQ(node1->pos, 0);
    EXPECT_EQ(node3->pos, 0);

    ASSERT_THAT(doc2.GetPos(), testing::ElementsAre(0, 0, 0));
}

TEST(Tree, instantiate) {
    // Firstly, instantiate a tree
    Tree tree(3, std::vector<double>{0.5, 0.5, 0.5});
    tree.Instantiate(tree.GetRoot(), 2);

    auto *root = tree.GetRoot();

    // Check tree size
    EXPECT_EQ(tree.NumNodes(0), 1);
    EXPECT_EQ(tree.NumNodes(1), 2);
    EXPECT_EQ(tree.NumNodes(2), 4);

    // Check stick-breaking score
    EXPECT_FLOAT_EQ(root->children[0]->sum_log_weight, log(1 / 1.5));
    EXPECT_FLOAT_EQ(root->children[1]->sum_log_weight, log((0.5 / 1.5) * (1 / 1.5)));

    // Add some observations
    auto *root_ch1 = root->children[1];
    auto *ch1_ch0 = root_ch1->children[0];

    Document doc;
    doc.c = decltype(doc.c){root, root_ch1, ch1_ch0};
    tree.UpdateNumDocs(doc.c.back(), 1);

    // Instantiate again
    tree.Instantiate(tree.GetRoot(), 2);

    EXPECT_EQ(tree.NumNodes(0), 1);
    EXPECT_EQ(tree.NumNodes(1), 3);
    EXPECT_EQ(tree.NumNodes(2), 7);

    EXPECT_EQ(root->children[0]->id, 2);

    EXPECT_FLOAT_EQ(root->children[0]->sum_log_weight, log(2 / 2.5));
    EXPECT_FLOAT_EQ(root->children[1]->sum_log_weight, log((0.5 / 2.5) * (1 / 1.5)));
    EXPECT_FLOAT_EQ(root->children[2]->sum_log_weight, log((0.5 / 2.5) * (0.5 / 1.5) * (1 / 1.5)));

    // Remove all instantiated nodes
    tree.Instantiate(tree.GetRoot(), 0);
    EXPECT_EQ(tree.GetAllNodes().size(), (size_t) 3);
}