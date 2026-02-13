from datetime import timedelta

from django.contrib.auth.models import User
from django.utils import timezone
from rest_framework import status
from rest_framework.test import APITestCase

from stations.models import Organization, OrganizationInvitation, OrganizationMember, Station


class StationAccessTests(APITestCase):
    def setUp(self):
        self.owner = User.objects.create_user(username="owner", email="owner@example.com", password="pass12345")
        self.other = User.objects.create_user(username="other", email="other@example.com", password="pass12345")
        self.org = Organization.objects.create(name="Org A", owner=self.owner)
        OrganizationMember.objects.create(
            organization=self.org,
            user=self.owner,
            role=OrganizationMember.ROLE_OWNER,
        )

    def test_member_can_create_station_in_own_org(self):
        self.client.force_authenticate(self.owner)

        response = self.client.post(
            "/api/stations/",
            {"name": "Station A", "org": self.org.id},
            format="json",
        )

        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertTrue(Station.objects.filter(org=self.org, name="Station A").exists())

    def test_non_member_cannot_create_station_in_foreign_org(self):
        self.client.force_authenticate(self.other)

        response = self.client.post(
            "/api/stations/",
            {"name": "Station A", "org": self.org.id},
            format="json",
        )

        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn("org", response.data)
        self.assertFalse(Station.objects.filter(org=self.org, name="Station A").exists())


class InvitationFlowTests(APITestCase):
    def setUp(self):
        self.owner = User.objects.create_user(username="owner", email="owner@example.com", password="pass12345")
        self.analyst = User.objects.create_user(username="analyst", email="analyst@example.com", password="pass12345")
        self.invited = User.objects.create_user(username="invitee", email="invitee@example.com", password="pass12345")

        self.org = Organization.objects.create(name="Org A", owner=self.owner)
        OrganizationMember.objects.create(
            organization=self.org,
            user=self.owner,
            role=OrganizationMember.ROLE_OWNER,
        )
        OrganizationMember.objects.create(
            organization=self.org,
            user=self.analyst,
            role=OrganizationMember.ROLE_ANALYST,
        )

    def test_owner_can_create_invitation_but_analyst_cannot(self):
        self.client.force_authenticate(self.owner)
        ok = self.client.post(
            f"/api/orgs/{self.org.id}/invitations/",
            {"invited_email": "new.user@example.com", "role": OrganizationMember.ROLE_VIEWER},
            format="json",
        )
        self.assertEqual(ok.status_code, status.HTTP_201_CREATED)

        self.client.force_authenticate(self.analyst)
        forbidden = self.client.post(
            f"/api/orgs/{self.org.id}/invitations/",
            {"invited_email": "another.user@example.com", "role": OrganizationMember.ROLE_VIEWER},
            format="json",
        )
        self.assertEqual(forbidden.status_code, status.HTTP_403_FORBIDDEN)

    def test_accept_invitation_requires_matching_email(self):
        invitation = OrganizationInvitation.objects.create(
            organization=self.org,
            invited_email="invitee@example.com",
            role=OrganizationMember.ROLE_ADMIN,
            invited_by=self.owner,
            expires_at=timezone.now() + timedelta(days=1),
        )

        self.client.force_authenticate(self.analyst)
        bad = self.client.post(
            "/api/orgs/invitations/accept/",
            {"token": invitation.token},
            format="json",
        )
        self.assertEqual(bad.status_code, status.HTTP_400_BAD_REQUEST)

        self.client.force_authenticate(self.invited)
        ok = self.client.post(
            "/api/orgs/invitations/accept/",
            {"token": invitation.token},
            format="json",
        )
        self.assertEqual(ok.status_code, status.HTTP_200_OK)

        invitation.refresh_from_db()
        self.assertEqual(invitation.status, OrganizationInvitation.STATUS_ACCEPTED)
        self.assertTrue(
            OrganizationMember.objects.filter(
                organization=self.org,
                user=self.invited,
                role=OrganizationMember.ROLE_ADMIN,
            ).exists()
        )
