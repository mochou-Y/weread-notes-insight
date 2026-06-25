import unittest

from src.app.main import build_psychological_report_lenses, navigation_pages


class AppPsychologicalReportTest(unittest.TestCase):
    def test_navigation_includes_new_report_page_without_replacing_noise_page(self):
        pages = navigation_pages()

        self.assertIn("🧠 心理画像报告", pages)
        self.assertIn("🔍 噪声洞察", pages)
        self.assertLess(pages.index("🧠 心理画像报告"), pages.index("🔍 噪声洞察"))

    def test_report_lenses_turn_analysis_into_judgment_first_sections(self):
        analysis = {
            "noise_stats": {"total": 12, "sub_clusters": 3},
            "user_profile": {
                "cognitive_style": {
                    "description": "你常从个体处境切入复杂系统问题。",
                    "keywords": ["系统", "个体", "判断"],
                },
                "knowledge_domains": [
                    {"domain": "技术社会", "weight": 0.32},
                    {"domain": "组织秩序", "weight": 0.21},
                ],
            },
            "bridge_patterns": [
                {"themes": ["技术", "人的处境"], "count": 8, "insight": "技术问题经常被你读成人的问题。"},
            ],
            "micro_themes": [{"label": "边缘探索", "size": 4}],
        }
        temporal = {
            "stability": {
                "core": [{"theme": "个体判断", "global_mean": 0.18}],
                "emerging": [{"theme": "技术治理", "first_half": 0.02, "second_half": 0.11}],
            },
            "narrative": {"shifts": "近期从自我理解转向技术与组织问题。"},
        }

        lenses = build_psychological_report_lenses(analysis, temporal)

        self.assertGreaterEqual(len(lenses), 4)
        self.assertEqual(lenses[0]["title"], "认知风格")
        self.assertIn("你常从个体处境切入复杂系统问题", lenses[0]["judgment"])
        self.assertTrue(all("judgment" in lens and "evidence" in lens for lens in lenses))


if __name__ == "__main__":
    unittest.main()
